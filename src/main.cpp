// Copyright (c) 2023 PhotonVision contributors

#include <cmath>
#include <iostream>
#include <optional>
#include <vector>
#include <iomanip>

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

using sleipnir::OptimizationProblem, sleipnir::Variable;
using VM = sleipnir::VariableMatrix;

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

void print_mat(VM &lhs, std::string_view name = "") {
  fmt::println("{} =", name);
  for (int i = 0; i < lhs.Rows(); i++) {
    for (int j = 0; j < lhs.Cols(); j++) {
      lhs(i, j).Update();
      fmt::print("{} ", lhs(i, j).Value());
    }
    fmt::println("");
  }
  fmt::println("");
}

VM divide_by_constant(VM &lhs, Variable rhs) {
  auto ret = VM(lhs.Rows(), lhs.Cols());
  for (int i = 0; i < lhs.Rows(); i++) {
    for (int j = 0; j < lhs.Cols(); j++) {
      ret(i, j) = lhs(i, j) / rhs;
    }
  }
  return ret;
}

VM add_constant(VM lhs, Variable &rhs) {
  auto ret = VM(lhs.Rows(), lhs.Cols());
  for (int i = 0; i < lhs.Rows(); i++) {
    for (int j = 0; j < lhs.Cols(); j++) {
      ret(i, j) = lhs(i, j) + rhs;
    }
  }
  return ret;
}

VM elementwise_divide(const VM &lhs, const VM &rhs) {
  assert(lhs.Rows() == rhs.Rows());
  assert(lhs.Cols() == rhs.Cols());

  VM ret(lhs.Rows(), lhs.Cols());
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
  VM worldToPixels(VM cameraToPoint) {
    auto X_c = cameraToPoint.Row(0);
    auto Y_c = cameraToPoint.Row(1);
    auto Z_c = cameraToPoint.Row(2);

    auto x_normalized = elementwise_divide(X_c, Z_c);
    auto y_normalized = elementwise_divide(Y_c, Z_c);

    VM u = add_constant(fx * VM(x_normalized), cx);
    VM v = add_constant(fy * VM(y_normalized), cy);

    VM ret(2, u.Cols());
    ret.Row(0) = u;
    ret.Row(1) = v;

    print_mat(ret, "uv");

    return ret;
  }
};

class CalibrationObjectView {
private:
  // Where we saw the corners at in the image
  Eigen::Matrix2Xd featureLocationsPixels;

  // Where the features are in 3d space on the object
  // std::vector<Point3d<double>> featureLocationsObjectSpace;
  Eigen::Matrix4Xd featureLocations;

  Transform<double> cameraToObjectGuess;

  // Decision variables for pose of this chessboard
public:
  VM t;
  VM r;

  CalibrationObjectView(Eigen::Matrix2Xd featureLocationsPixels_,
                        Eigen::Matrix4Xd featureLocations_,
                        Transform<double> cameraToObjectGuess_)
      : featureLocationsPixels{featureLocationsPixels_},
        featureLocations{featureLocations_},
        cameraToObjectGuess{cameraToObjectGuess_} {}

  Variable ReprojectionError(sleipnir::OptimizationProblem &problem,
                             CameraModel &model) {
    t = problem.DecisionVariable(3, 1);
    r = problem.DecisionVariable(3, 1);

    t(0, 0) = cameraToObjectGuess.t.x;
    t(1, 0) = cameraToObjectGuess.t.y;
    t(2, 0) = cameraToObjectGuess.t.z;
    r(0, 0) = cameraToObjectGuess.r.x;
    r(1, 0) = cameraToObjectGuess.r.y;
    r(2, 0) = cameraToObjectGuess.r.z;

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

    Variable theta = sleipnir::sqrt(r(0) * r(0) + r(1) * r(1) + r(2) * r(2));
    // TODO theta could be div-by-zero -- how do I deal with that?
    auto k = divide_by_constant(r, (theta + 1e-5));

    auto K = VM(3, 3);
    K(0, 0) = 0;
    K(0, 1) = -k(2);
    K(0, 2) = k(1);
    K(1, 0) = k(2);
    K(1, 1) = 0;
    K(1, 2) = -k(0);
    K(2, 0) = -k(1);
    K(2, 1) = k(0);
    K(2, 2) = 0;

    VM a = Eigen::Matrix<double, 3, 3>::Identity();
    VM b = K * sleipnir::sin(theta);
    VM c = K * K * (1 - sleipnir::cos(theta));

    auto R = a + b + c;

    // Homogenous transformation matrix from camera to object
    auto H = VM(4, 4);
    H.Block(0, 0, 3, 3) = R;
    H.Block(0, 3, 3, 1) = t;
    H.Block(3, 0, 1, 4) = (Eigen::Matrix4d() << 0, 0, 0, 1).finished();

    // Find where our chessboard features are in world space

    auto worldToCorners = H * featureLocations;

    print_mat(H, "H");
    std::cout << "featureLocations=\n" << featureLocations << "\n\n";
    print_mat(worldToCorners, "world2corners=H @ featureLocations");

    // And then project back to pixels
    auto pinholeProjectedPixels_model = model.worldToPixels(worldToCorners);
    auto reprojectionError_pixels =
        pinholeProjectedPixels_model - featureLocationsPixels;

    Variable cost = 0;
    for (int i = 0; i < reprojectionError_pixels.Rows(); i++) {
      for (int j = 0; j < reprojectionError_pixels.Cols(); j++) {
        cost += sleipnir::pow(reprojectionError_pixels(i, j), 2);
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
          double focalLengthGuess, double imageRows, double imageCols) {
  sleipnir::OptimizationProblem problem;

  CameraModel model{.fx = problem.DecisionVariable(),
                    .fy = problem.DecisionVariable(),
                    .cx = problem.DecisionVariable(),
                    .cy = problem.DecisionVariable()};

  model.fx = focalLengthGuess;
  model.fy = focalLengthGuess;
  model.cx = imageCols / 2.0;
  model.cy = imageRows / 2.0;

  Variable totalError = 0;
  for (auto &c : board_observations) {
    totalError += c.ReprojectionError(problem, model);
  }

  problem.Minimize(totalError);

  sleipnir::SolverConfig cfg;
  cfg.diagnostics = true;
  cfg.maxIterations = 100;

  auto stats = problem.Solve(cfg);

  fmt::print("fx = {}\n", model.fx.Value());
  fmt::print("fy = {}\n", model.fy.Value());
  fmt::print("cx = {}\n", model.cx.Value());
  fmt::print("cy = {}\n", model.cy.Value());

  int i = 0;
  for (auto &board : board_observations) {
    print_mat(board.t, fmt::format("board {} t", i));
    print_mat(board.r, fmt::format("board {} r", i));
    i++;
  }

  return std::nullopt;
}

int main() {

  std::string filename{"resources/corners.vnl"};
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
    for (int i = 0; i < 7; i++) {
      for (int j = 0; j < 7; j++) {
        featureLocations.col(i*7+j) << j * squareSize, i * squareSize, 0, 1;
        objectPoints3.push_back(Point3f(j * squareSize, i * squareSize, 0));
      }
    }

    // Initial guess at intrinsics
    Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 960/2, 0, 1000, 720/2, 0, 0, 1);
    Mat distCoeffs = Mat(4, 1, CV_64FC1, Scalar(0));

    Mat_<double> rvec, tvec;
    solvePnP(objectPoints3, imagePoints, cameraMatrix, distCoeffs, rvec, tvec,
            false, SOLVEPNP_EPNP);

    Transform<double> cameraToObject_bad_guess = {
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
  }
  // pixelLocations << 325.516, 132.934, 0.0, 371.214, 134.351, 0.0, 415.623,
  //     135.342, 0.0, 460.354, 136.823, 0.0, 504.145, 138.109, 0.0, 547.712,
  //     139.65, 0.0, 594.0, 148.683, 0.0, 324.871, 176.873, 0.0;

  Eigen::Matrix4Xd featureLocations(4, 8);
  featureLocations << 0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1;


  calibrate(board_views, 1000, 960, 720);

  return 0;

  // // The first input
  // fmt::print("u₀ = {}\n", U.Value(0, 0));
}
