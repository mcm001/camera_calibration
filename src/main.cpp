// Copyright (c) 2023 PhotonVision contributors

#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/problem.hpp>
#include <sleipnir/util/print.hpp>

#include "EigenFormat.hpp"
#include "camera_cal.hpp"

int main() {
  std::string filename{"./resources/corners_c920_1600_896.csv"};
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
  //   std::print("{}: ", k);
  //   for (const auto& thing : v) std::println(" -> x: {} y: {}", thing.x,
  //   thing.y); std::println("");
  // }

  std::vector<CalibrationObjectView> board_views;

  for (const auto &[k, v] : csvRows) {
    using namespace cv;

    Eigen::Matrix2Xd pixelLocations(2, v.size());
    std::vector<Point2f> imagePoints;

    size_t i = 0;
    for (const auto &corner : v) {
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
        featureLocations.col(i * 10 + j) << j * squareSize, i * squareSize, 0,
            1;
        objectPoints3.push_back(Point3f(j * squareSize, i * squareSize, 0));
      }
    }

    // Initial guess at intrinsics
    Mat cameraMatrix =
        (Mat_<double>(3, 3) << 1000, 0, 1600 / 2, 0, 1000, 896 / 2, 0, 0, 1);
    // Mat cameraMatrix = (Mat_<double>(3, 3) << 1.19060898e+03,
    // 0, 8.04278309e+02, 0, 1.19006900e+03, 4.55177360e+02, 0, 0, 1);
    Mat distCoeffs = Mat(4, 1, CV_64FC1, Scalar(0));

    Mat_<double> rvec, tvec;
    solvePnP(objectPoints3, imagePoints, cameraMatrix, distCoeffs, rvec, tvec,
             false, SOLVEPNP_EPNP);

    std::cout << "Rvec " << rvec << " tvec " << tvec << std::endl;

    Transform3d<double> cameraToObject_bad_guess = {
        .t = {tvec(0), tvec(1), tvec(2)},
        .r = {rvec(0), rvec(1), rvec(2)},
    };
    board_views.emplace_back(pixelLocations, featureLocations,
                             cameraToObject_bad_guess);

    // if (board_views.size() > 8) break;
  }

  // Solve with OpenCV
  if (1) {
    using namespace cv;

    std::vector<std::vector<Point3f>> objectPoints{};
    std::vector<std::vector<Point2f>> imagePoints{};
    Size imageSize(1600, 896);
    Mat cameraMatrix;
    Mat distCoeffs(1, 8, CV_64F);
    std::vector<Mat> rvecs, tvecs;

    // Copy data out of board_views (lol)
    for (const auto &view : board_views) {
      std::vector<Point3f> objPoints;
      std::vector<Point2f> imgPoints;

      const auto &featureLocs = view.featureLocations();
      const auto &pixelLocs = view.featureLocationsPixels();

      for (int i = 0; i < featureLocs.cols(); i++) {
        objPoints.emplace_back(featureLocs(0, i), // X
                               featureLocs(1, i), // Y
                               featureLocs(2, i)  // Z
        );

        imgPoints.emplace_back(pixelLocs(0, i), // u
                               pixelLocs(1, i)  // v
        );
      }

      objectPoints.push_back(objPoints);
      imagePoints.push_back(imgPoints);
    }

    double rms = ::cv::calibrateCamera(objectPoints, imagePoints, imageSize,
                                       cameraMatrix, distCoeffs, rvecs, tvecs,
                                       CALIB_USE_LU | CALIB_RATIONAL_MODEL);

    std::cout << "Camera matrix:\n" << cameraMatrix << std::endl;
    std::cout << "Distortion coefficients: " << distCoeffs << std::endl;
    std::cout << "RMS: " << rms << std::endl;
  } else {
    calibrate(board_views, 1, 2, 2);
    std::println("Expected fx=1000, fy=1000, cx=800, cy=448");
  }

  return 0;
}
