// Copyright (c) 2023 PhotonVision contributors

#include <cmath>
#include <optional>
#include <vector>

#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/time.h>

/*
Problem formuation:

we have N views of an object from a camera. We'll assume that the camera is
fixed at the world origin, and the object is moved around it. We run
unconstrained optimization to find a set of camera intrinsics and object poses
that minimize the reprojection error in pixels between predicted and observed
object feature locations, in pixels.

For each view, we
have:
- A feature location in pixels (u, v)
- A feature location in object 3d space (X, Y, Z)
- An initial guess at the transform between the fixed camera frame and the
object frame


We have these additional facts about the system:
- A camera model that unprojects a point in 3d space into a pixel location in
imager space. For now, let's assume this is an "OpenCV8" model, which encodes
camera focal length fₓ & f_y (in the same units as the feature locations above),
center location cₓ & c_y in pixels, and 8 distortion parameters (in OpenCV,
called k1, k2, p1, p2, k3, k4, k5, k6)
- An initial guess that this camera is zero-distortion and has an arbitrary
focal length
- The calibration board is also allowed to have a non-planer deformation in the
Z axis given by z=kₓ(1−x²)+k_y(1−y²), where x and y are distances between the
feature and the center of the calibration object.

Our calibration problem seeks to optimize the following variables.
- Camera to object transform
- Camera intrinsic parameters
- Object distortion

We represent the pose as a 6-d vector containing XYZ translation concatenated
with a Rodrigues-encoded rotation vector.

We also borrow mrcal's outlier rejection scheme. After each full optimization
run, all measurements with residuals greater than K standard deviations beyond 0
are marked as outliers, removed, and not considered for the next solve.

*/

using sleipnir::OptimizationProblem, sleipnir::Variable;

template <typename Number> struct Point2d {
  Number x;
  Number y;
};

template <typename Number> struct Point3d {
  Number x;
  Number y;
  Number z;
};

// rigid 6d transform, 3d translation + 3d Rodrigues rotation
template <typename Number> struct Transform {
  Point3d<Number> t;
  Point3d<Number> r;
};

sleipnir::VariableMatrix
elementwise_divide(const sleipnir::VariableMatrix &lhs,
                   const sleipnir::VariableMatrix &rhs) {
  assert(lhs.Rows() == rhs.Rows());
  assert(lhs.Cols() == rhs.Cols());

  sleipnir::VariableMatrix ret(lhs.Rows(), lhs.Cols());
  for (int i = 0; i < ret.Rows(); i++) {
    for (int j = 0; j < ret.Cols(); j++) {
      ret(i, j) = lhs(i, j) / rhs(i, j);
    }
  }

  return ret;
}

struct CameraModel {
  Variable fx;
  Variable fy;
  Variable cx;
  Variable cy;

  // TODO rename all the things
  using VM = sleipnir::VariableMatrix;
  VM worldToPixels(VM cameraToPoint) {
    auto X_c = cameraToPoint.Row(0);
    auto Y_c = cameraToPoint.Row(1);
    auto Z_c = cameraToPoint.Row(2);

    auto x_normalized = elementwise_divide(X_c, Z_c);
    auto y_normalized = elementwise_divide(Y_c, Z_c);

    VM u = fx * VM(x_normalized) + VM(cx);
    VM v = fy * VM(y_normalized) + VM(cy);

    VM ret(2, u.Cols());
    ret.Row(0) = u;
    ret.Row(1) = v;

    return ret;
  }
};

struct CalibrationObjectView {
  // Where we saw the corners at in the image
  Eigen::Matrix2Xd featureLocationsPixels;

  // Where the features are in 3d space on the object
  // std::vector<Point3d<double>> featureLocationsObjectSpace;
  Eigen::Matrix4Xd featureLocations;

  Transform<Variable> cameraToObject;

  Variable ReprojectionError(sleipnir::OptimizationProblem &problem,
                             CameraModel &model) {
    auto t = problem.DecisionVariable(3, 1);
    auto r = problem.DecisionVariable(3, 1);

    /*
    See: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula

    we want the homogonous transformation
    H = [ R | t ]
        [ 0 | 1 ]

    theta=norm(r),
    k = r/θ,
    k_x = r_x/theta, etc
    K = [
      0     -k_z  k_y
      k_z   0     -k_x
      -k_y  k_x   0
    ]
    where R = I(3, 3) + K * std::sin(θ) + K^2 (1-std::cos(θ)),
    */

    sleipnir::VariableMatrix theta =
        sleipnir::sqrt(r(0) * r(0) + r(1) * r(1) + r(2) * r(1));
    // TODO theta could be div-by-zero -- how do I deal with that?
    auto k = r / (theta + 1e-6);

    auto K = sleipnir::VariableMatrix(3, 3);
    K(0, 0) = 0;
    K(0, 1) = -k(2);
    K(0, 2) = k(1);
    K(1, 0) = k(2);
    K(1, 1) = 0;
    K(1, 2) = -k(0);
    K(2, 0) = -k(1);
    K(2, 1) = k(0);
    K(2, 2) = 0;

    auto R = Eigen::Matrix<double, 3, 3>::Identity() +
             K * sleipnir::sin(theta) + K * K * (1 - sleipnir::cos(theta));

    // Homogonous transformation matrix from camera to object
    auto H = sleipnir::VariableMatrix(4, 4);
    H.Block(0, 0, 3, 3) = R;
    H.Block(0, 3, 1, 3) = t;
    H.Block(3, 0, 1, 4) = (Eigen::Matrix4d() << 0, 0, 0, 1).finished();

    // Find where our chessboard features are in world space
    auto worldToCorners = H * featureLocations;

    // And then project back to pixels
    auto pinholeProjectedPixels_model = model.worldToPixels(worldToCorners);
    auto reprojectionError_pixels =
        pinholeProjectedPixels_model - featureLocationsPixels;

    Variable cost = 0;
    for (int i = 0; i < reprojectionError_pixels.Rows(); i++) {
      for (int j = 0; j < reprojectionError_pixels.Cols(); j++) {
        cost += sleipnir::Pow(reprojectionError_pixels(i, j), 2);
      }
    }

    return cost;
  }
};

struct CalibrationResult {
  std::vector<double> intrinsics;
  std::vector<double> residuals_pixels;
  Point2d<double> calobject_warp;
  double Noutliers;

  // final observations with optimized camera->object transforms
  std::vector<CalibrationObjectView> final_board_observations;
};

std::optional<CalibrationResult>
calibrate(std::vector<CalibrationObjectView> board_observations,
          double focalLengthGuess);

int main() {
  // Stuff I copy pasted from a Sleipnir example -- not relevant

  constexpr auto T = 5_s;
  constexpr units::second_t dt = 5_ms;
  constexpr int N = T / dt;

  // Flywheel model:
  // States: [velocity]
  // Inputs: [voltage]
  Eigen::Matrix<double, 1, 1> A{std::exp(-dt.value())};
  Eigen::Matrix<double, 1, 1> B{1.0 - std::exp(-dt.value())};

  sleipnir::OptimizationProblem problem;
  auto X = problem.DecisionVariable(1, N + 1);
  auto U = problem.DecisionVariable(1, N);

  // Dynamics constraint
  for (int k = 0; k < N; ++k) {
    problem.SubjectTo(X.Col(k + 1) == A * X.Col(k) + B * U.Col(k));
  }

  // State and input constraints
  problem.SubjectTo(X.Col(0) == 0.0);
  problem.SubjectTo(-12 <= U);
  problem.SubjectTo(U <= 12);

  // Cost function - minimize error
  Eigen::Matrix<double, 1, 1> r{10.0};
  sleipnir::Variable J = 0.0;
  for (int k = 0; k < N + 1; ++k) {
    J += (r - X.Col(k)).T() * (r - X.Col(k));
  }
  problem.Minimize(J);

  problem.Solve();

  // The first state
  fmt::print("x₀ = {}\n", X.Value(0, 0));

  // The first input
  fmt::print("u₀ = {}\n", U.Value(0, 0));
}
