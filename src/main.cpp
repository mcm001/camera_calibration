// Copyright (c) 2023 PhotonVision contributors

#include <cmath>

#include <sleipnir/optimization/OptimizationProblem.hpp>
#include <units/time.h>

/*
Problem formuation:

we have N views of an object from a camera. We'll assume that the camera is
fixed at the world origin, and the object is moved around it. We run unconstrained optimization to find a set of camera intrinsics and object poses that minimize the reprojection error in pixels between predicted and observed object feature locations, in pixels.

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
with a Rodriguez-encoded rotation vector.

*/

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
