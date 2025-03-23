#pragma once

#include <Eigen/Core>
#include <optional>
#include <vector>
#include <concepts>
#include <sleipnir/autodiff/variable.hpp>
#include <sleipnir/autodiff/variable_matrix.hpp>
#include <sleipnir/optimization/problem.hpp>

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
 * Rigid 6D transform, where r is a Rodrigues rotation vector.
 */
template <std::floating_point T>
struct Transform3d {
    Point3d<T> t;  // 3D translation
    Point3d<T> r;  // 3D rotation vector
};

struct CameraModel {
    slp::Variable fx;
    slp::Variable fy;
    slp::Variable cx;
    slp::Variable cy;

    explicit CameraModel(slp::Problem& problem);
    slp::VariableMatrix WorldToPixels(const slp::VariableMatrix& cameraToPoint) const;
};

class CalibrationObjectView {
public:
    slp::VariableMatrix t;  // Translation of chessboard
    slp::VariableMatrix r;  // Rotation of chessboard

    CalibrationObjectView(Eigen::Matrix2Xd featureLocationsPixels,
                         Eigen::Matrix4Xd featureLocations,
                         Transform3d<double> cameraToObjectGuess);

    slp::Variable ReprojectionError(slp::Problem& problem, const CameraModel& model);
    const Eigen::Matrix2Xd& featureLocationsPixels() const;
    const Eigen::Matrix4Xd& featureLocations() const;

private:
    Eigen::Matrix2Xd m_featureLocationsPixels;
    Eigen::Matrix4Xd m_featureLocations;
    Transform3d<double> m_cameraToObjectGuess;
};

struct CalibrationResult {
    std::vector<double> intrinsics;
    std::vector<double> residuals_pixels;
    Point2d<double> calobject_warp;
    double Noutliers;
    std::vector<CalibrationObjectView> final_boardObservations;
};

slp::VariableMatrix mat_of(int rows, int cols, int value);

std::optional<CalibrationResult> calibrate(
    std::vector<CalibrationObjectView> boardObservations,
    double focalLengthGuess,
    double imageCols,
    double imageRows);