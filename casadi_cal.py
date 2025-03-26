from dataclasses import dataclass
import math
from typing import List
import casadi as cs
from wpimath.geometry import Pose3d, Translation3d, Rotation3d, Quaternion

@dataclass
class Point3:
    x: float
    y: float
    z: float

@dataclass
class Point2:
    x: float
    y: float

@dataclass
class Observation:
    locationInObjectSpace: List[Point3]
    locationInImageSpace: List[Point2]
    snapshotName: str
    optimizedCameraToObject: Pose3d = None

def load_json():
    import json
    with open('resources/photon_calibration_Microsoft_LifeCam_HD-3000_1280x720.json') as f:
        data = json.load(f)
    
    observations = []
    for obs in data['observations']:
        obj_points = [Point3(p['x'], p['y'], p['z']) for p in obs['locationInObjectSpace']]
        img_points = [Point2(p['x'], p['y']) for p in obs['locationInImageSpace']]
        optimizedCameraToObject = Pose3d(
            Translation3d(
                x=obs['optimisedCameraToObject']['translation']['x'],
                y=obs['optimisedCameraToObject']['translation']['y'],
                z=obs['optimisedCameraToObject']['translation']['z']
            ),
            Rotation3d(Quaternion(
                w=obs['optimisedCameraToObject']['rotation']['quaternion']['W'],
                x=obs['optimisedCameraToObject']['rotation']['quaternion']['X'],
                y=obs['optimisedCameraToObject']['rotation']['quaternion']['Y'],
                z=obs['optimisedCameraToObject']['rotation']['quaternion']['Z']
            ))
        )
        assert len(obj_points) == len(img_points)
        observations.append(Observation(obj_points, img_points, obs['snapshotName'], 
                                        optimizedCameraToObject))
    
    return observations

def calculate_reprojection_error(opti, obj_points, img_points, fx, fy, cx, cy):
    """
    Calculate reprojection error for a single calibration board observation
    
    Args:
        opti: cs.Opti() optimizer instance
        obj_points: List of 3D points in object space
        img_points: List of 2D points in image space
        fx, fy: Focal lengths
        cx, cy: Principal point coordinates
    """
    # Create optimization variables for translation and rotation
    t = opti.variable(3, 1)
    r = opti.variable(3, 1)  # Rotation vector (Rodriguez)

    # Initial guess - board 1m away with no rotation. probably a pretty bad guess
    opti.set_initial(t, [0, 0, 1])
    opti.set_initial(r, [0, 0, 0])

    # Convert object points to Casadi matrix
    obj_points_mat = cs.DM.zeros(4, len(obj_points))
    for i, p in enumerate(obj_points):
        obj_points_mat[0:3, i] = [p.x, p.y, p.z]
        obj_points_mat[3, i] = 1.0

    # Convert image points to Casadi matrix
    img_points_mat = cs.DM.zeros(2, len(img_points))
    for i, p in enumerate(img_points):
        img_points_mat[:, i] = [p.x, p.y]

    # Calculate rotation matrix using Rodriguez formula
    theta = cs.sqrt(r[0]**2 + r[1]**2 + r[2]**2 + 1e-8)
    k = r / (theta + 1e-8)  # Normalized rotation axis
    
    # Skew symmetric matrix K
    K = cs.vertcat(
        cs.horzcat(0, -k[2], k[1]),
        cs.horzcat(k[2], 0, -k[0]),
        cs.horzcat(-k[1], k[0], 0)
    )
    
    # Rotation matrix
    R = cs.DM.eye(3) + K * cs.sin(theta) + K @ K * (1 - cs.cos(theta))

    # Homogeneous transformation matrix
    H = cs.vertcat(
        cs.horzcat(R, t),
        cs.horzcat(cs.DM.zeros(1, 3), cs.DM.ones(1, 1))
    )

    # Transform points to camera space
    points_camera = H @ obj_points_mat

    # Project to image plane
    x_normalized = points_camera[0, :] / points_camera[2, :]
    y_normalized = points_camera[1, :] / points_camera[2, :]
    
    # Apply camera intrinsics
    projected_points = cs.vertcat(
        fx * x_normalized + cx,
        fy * y_normalized + cy
    )

    # Calculate reprojection error
    error = projected_points - img_points_mat
    cost = cs.sum1(cs.sum2(error**2))

    return cost, t, r

data = load_json()

opti = cs.Opti()

# Create camera intrinsics variables
fx = opti.variable()
fy = opti.variable()
cx = opti.variable()
cy = opti.variable()

# with an initial guess
opti.set_initial(fx, 1000)
opti.set_initial(fy, 1000)
opti.set_initial(cx, 1280/2)
opti.set_initial(cy, 720/2)

tvecs = []
rvecs = []

total_cost = 0
for obs in data:
    cost, t, r = calculate_reprojection_error(opti, obs.locationInObjectSpace, obs.locationInImageSpace, fx, fy, cx, cy)

    total_cost += cost
    tvecs.append(t)
    rvecs.append(r)

    if len(tvecs) > 6:
        break

opti.minimize(total_cost)

opti.solver('ipopt')
sol = opti.solve()

print(f"Final error: {math.sqrt(sol.value(total_cost))} pixels")

# print optimized tvecs and rvecs
for t, r, obs in zip(tvecs, rvecs, data):
    print(f"t: {sol.value(t)}, r: {sol.value(r)}")
    print(f"   Expected: {obs.optimizedCameraToObject.translation()}, {obs.optimizedCameraToObject.rotation().axis() * obs.optimizedCameraToObject.rotation().angle}")
    print("")

# And print camera calibration values
print(f"fx: {sol.value(fx)}")
print(f"fy: {sol.value(fy)}")
print(f"cx: {sol.value(cx)}")
print(f"cy: {sol.value(cy)}")

print(
"""
For reference, we previously calculated the following values: 
1132.983599412085  fx
1138.2884596791835 fy
610.3195830765223  cx
346.4121207400337  cy
"""
)
