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

#include <nlohmann/json.hpp>

struct Observation {
  std::string snapshotName;
  std::vector<cv::Point2f> locationInImageSpace;
  std::vector<cv::Point3f> locationInObjectSpace;
};

std::map<std::string, Observation> parseJson() {
    std::string filename{"./resources/photon_calibration_Microsoft_LifeCam_HD-3000_1280x720.json"};
    std::ifstream input{filename};
    nlohmann::json j;
    input >> j;

    std::map<std::string, Observation> jsonRows;
    
    for (const auto& snapshot : j["observations"]) {
        std::string snapshotId = snapshot["snapshotName"];
        std::vector<cv::Point2f> imgPoints;
        std::vector<cv::Point3f> objPoints;
        
        for (const auto& point : snapshot["locationInImageSpace"]) {
            imgPoints.push_back({
                point["x"].get<double>(),
                point["y"].get<double>()
            });
        }
        for (const auto& point : snapshot["locationInObjectSpace"]) {
            objPoints.push_back({
                point["x"].get<double>(),
                point["y"].get<double>(),
                point["z"].get<double>()
            });
        }
        
        jsonRows[snapshotId] = Observation{
            .snapshotName = snapshotId,
            .locationInImageSpace = imgPoints,
            .locationInObjectSpace = objPoints
        };
    }

    return jsonRows;
}

std::map<std::string, std::vector<Point2d<double>>> parseCsv() {
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

  return csvRows;
}

int main() {
  int NUM_ROWS = 7;
  int NUM_COLS = 7;
  const double squareSize = 0.0254;
  cv::Size imageSize(1280, 720);
  

  auto csvRows {parseJson()};

  // debug print to verify we parsed things right
  // for (const auto& [k, v] : csvRows) {
  //   std::print("{}: ", k);
  //   for (const auto& thing : v) std::println(" -> x: {} y: {}", thing.x,
  //   thing.y); std::println("");
  // }

  std::vector<CalibrationObjectView> board_views;

  // for (const auto &[k, v] : csvRows) {
  //   using namespace cv;
  //   Eigen::Matrix2Xd pixelLocations(2, v.size());
  //   std::vector<Point2f> imagePoints;
  //   size_t i = 0;
  //   for (const auto &corner : v) {
  //     pixelLocations.col(i) << corner.x, corner.y;
  //     imagePoints.emplace_back(corner.x, corner.y);
  //     ++i;
  //   }
  //   Eigen::Matrix4Xd featureLocations(4, v.size());
  //   std::vector<Point3f> objectPoints3;
  //   // Fill in object/image points
  //   for (int i = 0; i < NUM_ROWS; i++) {
  //     for (int j = 0; j < NUM_COLS; j++) {
  //       featureLocations.col(i * 10 + j) << j * squareSize, i * squareSize, 0,
  //           1;
  //       objectPoints3.push_back(Point3f(j * squareSize, i * squareSize, 0));
  //     }
  //   }
  //   // Initial guess at intrinsics
  //   Mat cameraMatrix =
  //       (Mat_<double>(3, 3) << 1000, 0, imageSize.width / 2, 0, 1000, imageSize.height / 2, 0, 0, 1);
  //   // Mat cameraMatrix = (Mat_<double>(3, 3) << 1.19060898e+03,
  //   // 0, 8.04278309e+02, 0, 1.19006900e+03, 4.55177360e+02, 0, 0, 1);
  //   Mat distCoeffs = Mat(4, 1, CV_64FC1, Scalar(0));
  //   Mat_<double> rvec, tvec;
  //   solvePnP(objectPoints3, imagePoints, cameraMatrix, distCoeffs, rvec, tvec,
  //            false, SOLVEPNP_EPNP);
  //   std::cout << "Rvec " << rvec << " tvec " << tvec << std::endl;
  //   Transform3d<double> cameraToObject_bad_guess = {
  //       .t = {tvec(0), tvec(1), tvec(2)},
  //       .r = {rvec(0), rvec(1), rvec(2)},
  //   };
  //   board_views.emplace_back(pixelLocations, featureLocations,
  //                            cameraToObject_bad_guess);
  //   // if (board_views.size() > 8) break;
  // }

  // Solve with OpenCV
  if (1) {
    using namespace cv;

    std::vector<std::vector<cv::Point3f>> objectPoints{};
    std::vector<std::vector<cv::Point2f>> imagePoints{};

    // turn csvRows into object and image points
    for (const auto &[k, v] : csvRows) {
      objectPoints.push_back(v.locationInObjectSpace);
      imagePoints.push_back(v.locationInImageSpace);
    }

    std::println("Saw {} images", objectPoints.size());
    std::println("In each observation, we saw:");
    std::for_each(objectPoints.begin(), objectPoints.end(), [](const auto& v) {
      std::println("  {} points", v.size());
    });

    Mat cameraMatrix;
    Mat distCoeffs(1, 8, CV_64F);
    std::vector<Mat> rvecs, tvecs;


    double rms = ::cv::calibrateCamera(objectPoints, imagePoints, imageSize,
                                       cameraMatrix, distCoeffs, rvecs, tvecs,
                                       CALIB_USE_LU);

    std::cout << "Camera matrix:\n" << cameraMatrix << std::endl;
    std::cout << "Distortion coefficients: " << distCoeffs << std::endl;
    std::cout << "RMS: " << rms << std::endl;
  } else {
    calibrate(board_views, 1, 2, 2);
    std::println("Expected fx=1000, fy=1000, cx=800, cy=448");
  }

  return 0;
}
