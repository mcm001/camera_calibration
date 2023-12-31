// Copyright (c) 2023 PhotonVision contributors

#include <cmath>
#include <iostream>
#include <optional>
#include <vector>
#include <iomanip>

#include <sleipnir/Formatters.hpp>

#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/time.h>

#include <opencv4/opencv2/opencv.hpp>

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

template <std::floating_point T>
struct Point2d {
  T x;
  T y;
};

template <std::floating_point T>
struct Point3d {
  T x;
  T y;
  T z;
};

/**
 * Rigid 6D transform.
 */
template <std::floating_point T>
struct Transform3d {
  /// 3D translation
  Point3d<T> t;
  /// 3D rotation vector
  Point3d<T> r;
};

namespace slp = sleipnir;

slp::VariableMatrix mat_of(int rows, int cols, int value) {
  auto ret = slp::VariableMatrix(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      ret(i, j) = value;
    }
  }
  return ret;
}

struct CameraModel {
  slp::Variable fx;
  slp::Variable fy;
  slp::Variable cx;
  slp::Variable cy;

  explicit CameraModel(slp::OptimizationProblem& problem)
      : fx{problem.DecisionVariable()},
        fy{problem.DecisionVariable()},
        cx{problem.DecisionVariable()},
        cy{problem.DecisionVariable()} {}

  // TODO: Rename all the things
  slp::VariableMatrix WorldToPixels(
      const slp::VariableMatrix& cameraToPoint) const {
    auto X_c = cameraToPoint.Row(0);
    auto Y_c = cameraToPoint.Row(1);
    auto Z_c = cameraToPoint.Row(2);

    auto x_normalized = slp::CwiseReduce(X_c, Z_c, std::divides<>{});
    auto y_normalized = slp::CwiseReduce(Y_c, Z_c, std::divides<>{});

    slp::VariableMatrix u =
        fx * x_normalized +
        slp::VariableMatrix{cx} * Eigen::RowVectorXd::Ones(x_normalized.Cols());
    slp::VariableMatrix v =
        fy * y_normalized +
        slp::VariableMatrix{cy} * Eigen::RowVectorXd::Ones(y_normalized.Cols());

    return slp::Block({{u}, {v}});
  }
};

class CalibrationObjectView {
 public:
  /// Translation of chessboard
  slp::VariableMatrix t;

  /// Rotation of chessboard
  slp::VariableMatrix r;

  CalibrationObjectView(Eigen::Matrix2Xd featureLocationsPixels,
                        Eigen::Matrix4Xd featureLocations,
                        Transform3d<double> cameraToObjectGuess)
      : m_featureLocationsPixels{std::move(featureLocationsPixels)},
        m_featureLocations{std::move(featureLocations)},
        m_cameraToObjectGuess{std::move(cameraToObjectGuess)} {}

  slp::Variable ReprojectionError(slp::OptimizationProblem& problem,
                                  const CameraModel& model) {
    // t = problem.DecisionVariable(3);
    t = slp::VariableMatrix(3, 1);
    t(0).SetValue(m_cameraToObjectGuess.t.x);
    t(1).SetValue(m_cameraToObjectGuess.t.y);
    t(2).SetValue(m_cameraToObjectGuess.t.z);

    // r = problem.DecisionVariable(3);
    r = slp::VariableMatrix(3, 1);
    r(0).SetValue(m_cameraToObjectGuess.r.x);
    r(1).SetValue(m_cameraToObjectGuess.r.y);
    r(2).SetValue(m_cameraToObjectGuess.r.z);

    // See: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    //
    // We want the homogonous transformation:
    //
    //   H = [R  t]
    //       [0  1]
    //
    //   θ = norm(r)
    //   k = r/θ
    //
    //   k_x = r_x/θ
    //   k_y = r_y/θ
    //   k_z = r_y/θ
    //
    //       [  0   −k_z   k_y]
    //   K = [ k_z    0   −k_x]
    //       [−k_y   k_x    0 ]
    //
    // where R = I₃ₓ₃ + K std::sin(θ) + K²(1 − std::cos(θ))

    slp::Variable theta = slp::hypot(r(0), r(1), r(2));
    auto k = r / slp::VariableMatrix{theta + 1e-5};

    slp::VariableMatrix K{{0, -k(2), k(1)}, {k(2), 0, -k(0)}, {-k(1), k(0), 0}};

    auto R = Eigen::Matrix<double, 3, 3>::Identity() + K * slp::sin(theta) +
             K * K * (1 - slp::cos(theta));

    // Homogenous transformation matrix from camera to object
    auto H = slp::Block({{R, t}, {slp::VariableMatrix{{0, 0, 0, 1}}}});

    // Find where our chessboard features are in world space
    auto worldToCorners = H * m_featureLocations;

    fmt::print("H =\n{}\n", H);
    // fmt::print("featureLocations=\n{}\n", m_featureLocations);
    // fmt::print("world2corners = H @ featureLocations =\n{}\n", worldToCorners);

    // And then project back to pixels
    auto pinholeProjectedPixels_model = model.WorldToPixels(worldToCorners);

    fmt::println("Projected pixel locations:\n{}", pinholeProjectedPixels_model.Block(0, 0, 2, 12));
    fmt::println("Observed locations:\n{}", m_featureLocationsPixels.block(0, 0, 2, 12));

    auto reprojectionError_pixels =
        pinholeProjectedPixels_model - m_featureLocationsPixels;
    fmt::println("Reprojection error:\n{}", reprojectionError_pixels.Block(0, 0, 2, 12));

    slp::Variable cost = 0.0;
    for (int i = 0; i < reprojectionError_pixels.Rows(); ++i) {
      for (int j = 0; j < reprojectionError_pixels.Cols(); ++j) {
        cost += slp::pow(reprojectionError_pixels(i, j), 2);
      }
    }

    return cost;
  }

 private:
  // Where we saw the corners at in the image
  Eigen::Matrix2Xd m_featureLocationsPixels;

  // Where the features are in 3d space on the object
  // std::vector<Point3d<double>> featureLocationsObjectSpace;
  Eigen::Matrix4Xd m_featureLocations;

  Transform3d<double> m_cameraToObjectGuess;
};

struct CalibrationResult {
  std::vector<double> intrinsics;
  std::vector<double> residuals_pixels;
  Point2d<double> calobject_warp;
  double Noutliers;

  // Final observations with optimized camera->object transforms
  std::vector<CalibrationObjectView> final_boardObservations;
};

std::optional<CalibrationResult> calibrate(
    std::vector<CalibrationObjectView> boardObservations,
    double focalLengthGuess, double imageCols, double imageRows) {
  slp::OptimizationProblem problem;

  CameraModel model{problem};

  model.fx.SetValue(focalLengthGuess);
  model.fy.SetValue(focalLengthGuess);
  model.cx.SetValue(imageCols / 2.0);
  model.cy.SetValue(imageRows / 2.0);

  slp::Variable cost = 0.0;
  for (auto& c : boardObservations) {
    cost += c.ReprojectionError(problem, model);
  }
  problem.Minimize(cost);

  problem.Callback([](const slp::SolverIterationInfo& info) {
    // fmt::print("x =\n{}\n", info.x);
    // fmt::print("Hessian =\n{}\n", info.H);
    // fmt::print("gradient =\n{}\n", info.g);

    return false;
  });

  fmt::println("Prior:");
  fmt::print("fx = {}\n", model.fx.Value());
  fmt::print("fy = {}\n", model.fy.Value());
  fmt::print("cx = {}\n", model.cx.Value());
  fmt::print("cy = {}\n", model.cy.Value());

  problem.Solve({.tolerance=1e-10, .diagnostics = true});

  fmt::println("Final:");
  fmt::print("fx = {}\n", model.fx.Value());
  fmt::print("fy = {}\n", model.fy.Value());
  fmt::print("cx = {}\n", model.cx.Value());
  fmt::print("cy = {}\n", model.cy.Value());

  int i = 0;
  for (auto& board : boardObservations) {
    fmt::print("board {} t =\n{}\n", i, board.t(0,0).Value());
    fmt::print("board {} r =\n{}\n", i, board.r(0,0).Value());
    ++i;
  }

  return std::nullopt;
}

int main() {

  std::string filename{"/home/matt/camera_calibration/resources/corners_c920_1600_896.csv"};
  std::ifstream input{filename};

  std::map<std::string, std::vector<Point2d<double>>> csvRows;

  for (std::string line; std::getline(input, line);) {
    std::istringstream ss(std::move(line));
    Point2d<double> row;

    // std::getline can split on other characters, here we use ' '
    std::string imageName;
    ss >> imageName >> row.x >> row.y;

    if (csvRows.find(imageName) == csvRows.end()) {
      csvRows.insert({imageName, std::vector<Point2d<double>>()});
    }

    csvRows.at(imageName).push_back(row);
  }

  // debug print to verify we parsed things right
  // for (const auto& [k, v] : csvRows) {
  //   fmt::print("{}: ", k);
  //   for (const auto& thing : v) fmt::println(" -> x: {} y: {}", thing.x, thing.y);
  //   fmt::println("");
  // }

  std::vector<CalibrationObjectView> board_views;

  for (const auto& [k, v] : csvRows) {
    using namespace cv;

    Eigen::Matrix2Xd pixelLocations(4, v.size());
    std::vector<Point2f> imagePoints;

    size_t i = 0;
    for (const auto& corner : v) {
      pixelLocations.col(i) << corner.x, corner.y;
      imagePoints.emplace_back(corner.x, corner.y);
      ++i;
    }

    Eigen::Matrix4Xd featureLocations(4, v.size());
    std::vector<Point3f> objectPoints3;

    const double squareSize = 0.0254;

    // Fill in object/image points
    // pre-knowledge -- 49 corners
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        featureLocations.col(i*10+j) << j * squareSize, i * squareSize, 0, 1;
        objectPoints3.push_back(Point3f(j * squareSize, i * squareSize, 0));
      }
    }

    // Initial guess at intrinsics
    Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 1600/2, 0, 1000, 896/2, 0, 0, 1);
    // Mat cameraMatrix = (Mat_<double>(3, 3) << 1.19060898e+03, 0, 8.04278309e+02, 0, 1.19006900e+03, 4.55177360e+02, 0, 0, 1);
    Mat distCoeffs = Mat(4, 1, CV_64FC1, Scalar(0));

    Mat_<double> rvec, tvec;
    solvePnP(objectPoints3, imagePoints, cameraMatrix, distCoeffs, rvec, tvec,
            false, SOLVEPNP_EPNP);

    std::cout << "Rvec " << rvec << " tvec " << tvec <<std::endl;

    Transform3d<double> cameraToObject_bad_guess = {
      .t = {
        tvec(0), tvec(1), tvec(2)
      },
      .r = {
        rvec(0), rvec(1), rvec(2)
      },
    };
    board_views.emplace_back(
      pixelLocations, featureLocations, cameraToObject_bad_guess
    );

    break;
  }

  calibrate(board_views, 1000, 1600, 896);

  return 0;

  // // The first input
  // fmt::print("u₀ = {}\n", U.Value(0, 0));
}
