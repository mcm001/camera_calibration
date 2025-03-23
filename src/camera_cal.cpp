// Copyright (c) 2023 PhotonVision contributors

#include "camera_cal.hpp"

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

slp::VariableMatrix mat_of(int rows, int cols, int value) {
  auto ret = slp::VariableMatrix(rows, cols);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      ret(i, j) = value;
    }
  }
  return ret;
}

CameraModel::CameraModel(slp::Problem &problem)
    : fx{problem.decision_variable()}, fy{problem.decision_variable()},
      cx{problem.decision_variable()}, cy{problem.decision_variable()} {}

// TODO: Rename all the things
slp::VariableMatrix
CameraModel::WorldToPixels(const slp::VariableMatrix &cameraToPoint) const {
  auto X_c = cameraToPoint.row(0);
  auto Y_c = cameraToPoint.row(1);
  auto Z_c = cameraToPoint.row(2);

  auto x_normalized = slp::cwise_reduce(X_c, Z_c, std::divides<>{});
  auto y_normalized = slp::cwise_reduce(Y_c, Z_c, std::divides<>{});

  slp::VariableMatrix u =
      fx * x_normalized +
      slp::VariableMatrix{cx} * Eigen::RowVectorXd::Ones(x_normalized.cols());
  slp::VariableMatrix v =
      fy * y_normalized +
      slp::VariableMatrix{cy} * Eigen::RowVectorXd::Ones(y_normalized.cols());

  return slp::block({{u}, {v}});
}

CalibrationObjectView::CalibrationObjectView(
    Eigen::Matrix2Xd featureLocationsPixels, Eigen::Matrix4Xd featureLocations,
    Transform3d<double> cameraToObjectGuess)
    : m_featureLocationsPixels{std::move(featureLocationsPixels)},
      m_featureLocations{std::move(featureLocations)},
      m_cameraToObjectGuess{std::move(cameraToObjectGuess)} {}

slp::Variable
CalibrationObjectView::ReprojectionError(slp::Problem &problem,
                                         const CameraModel &model) {
  // t = problem.decision_variable(3);
  t = slp::VariableMatrix(3, 1);
  t[0].set_value(m_cameraToObjectGuess.t.x);
  t[1].set_value(m_cameraToObjectGuess.t.y);
  t[2].set_value(m_cameraToObjectGuess.t.z);

  // r = problem.decision_variable(3);
  r = slp::VariableMatrix(3, 1);
  r[0].set_value(m_cameraToObjectGuess.r.x);
  r[1].set_value(m_cameraToObjectGuess.r.y);
  r[2].set_value(m_cameraToObjectGuess.r.z);

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

  slp::Variable theta = slp::hypot(r[0], r[1], r[2]);
  auto k = r / slp::Variable{theta + 1e-5}; // avoid division by zero

  slp::VariableMatrix K{{0, -k[2], k[1]}, {k[2], 0, -k[0]}, {-k[1], k[0], 0}};

  auto R = Eigen::Matrix<double, 3, 3>::Identity() + K * slp::sin(theta) +
           K * K * (1 - slp::cos(theta));

  // Homogenous transformation matrix from camera to object
  auto H = slp::block({{R, t}, {slp::VariableMatrix{{0, 0, 0, 1}}}});

  // Find where our chessboard features are in world space
  auto worldToCorners = H * m_featureLocations;

  std::print("H =\n{}\n", H.value());
  // std::print("featureLocations=\n{}\n", m_featureLocations);
  // std::print("world2corners = H @ featureLocations =\n{}\n",
  // worldToCorners);

  // And then project back to pixels
  auto pinholeProjectedPixels_model = model.WorldToPixels(worldToCorners);

  std::println("Projected pixel locations:\n{}",
               pinholeProjectedPixels_model.block(0, 0, 2, 12).value());
  std::println("Observed locations:\n{}",
               m_featureLocationsPixels.block(0, 0, 2, 12));

  auto reprojectionError_pixels =
      pinholeProjectedPixels_model - m_featureLocationsPixels;
  std::println("Reprojection error:\n{}",
               reprojectionError_pixels.block(0, 0, 2, 12).value());

  slp::Variable cost = 0.0;
  for (int i = 0; i < reprojectionError_pixels.rows(); ++i) {
    for (int j = 0; j < reprojectionError_pixels.cols(); ++j) {
      cost += slp::pow(reprojectionError_pixels(i, j), 2);
    }
  }

  return cost;
}

const Eigen::Matrix2Xd &CalibrationObjectView::featureLocationsPixels() const {
  return m_featureLocationsPixels;
}

const Eigen::Matrix4Xd &CalibrationObjectView::featureLocations() const {
  return m_featureLocations;
}

std::optional<CalibrationResult>
calibrate(std::vector<CalibrationObjectView> boardObservations,
          double focalLengthGuess, double imageCols, double imageRows) {
  slp::Problem problem;

  CameraModel model{problem};

  model.fx.set_value(focalLengthGuess);
  model.fy.set_value(focalLengthGuess);
  model.cx.set_value(imageCols / 2.0);
  model.cy.set_value(imageRows / 2.0);

  slp::Variable cost = 0.0;
  for (auto &c : boardObservations) {
    cost += c.ReprojectionError(problem, model);
  }
  problem.minimize(cost);

  problem.add_callback([](const slp::IterationInfo &info) {
    // std::print("x =\n{}\n", info.x);
    // std::print("Hessian =\n{}\n", info.H);
    // std::print("gradient =\n{}\n", info.g);

    return false;
  });

  std::println("Prior:");
  std::print("fx = {}\n", model.fx.value());
  std::print("fy = {}\n", model.fy.value());
  std::print("cx = {}\n", model.cx.value());
  std::print("cy = {}\n", model.cy.value());

  std::println("Initial cost = {}", cost.value());

  problem.solve({.tolerance = 1e-7, .max_iterations = 25, .diagnostics = true});

  std::println("Final:");
  std::println("cost = {}", cost.value());
  std::print("fx = {}\n", model.fx.value());
  std::print("fy = {}\n", model.fy.value());
  std::print("cx = {}\n", model.cx.value());
  std::print("cy = {}\n", model.cy.value());

  // int i = 0;
  // for (auto &board : boardObservations) {
  //   std::print("board {} t =\n{}\n", i, board.t(0, 0).value());
  //   std::print("board {} r =\n{}\n", i, board.r(0, 0).value());
  //   ++i;
  // }

  return std::nullopt;
}
