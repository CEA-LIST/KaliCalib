import numpy as np
import cv2
from deepsport_utilities.calib import Calib, Point2D, Point3D


MEAN_H = [
    [1.94421059e03, -2.34553735e02, 1.61330395e02, -1.89485410e06],
    [-1.72322461e01, 5.88543639e02, 1.93678628e03, 2.69150981e05],
    [4.89042879e-02, -8.25861073e-01, 4.17496411e-01, 2.73299170e03],
]


def compute_camera_model(points2d, points3d, output_shape):

    height, width = output_shape
    h = np.eye(3, 4)
    h[2, 3] = 1.0
    calib = Calib.from_P(np.array(MEAN_H), width=width, height=height)
    # cv2 calibrateCamera requires at least 4 points
    if len(points2d) > 5:
        points2d_ = Point2D(np.array(points2d).T)
        points3d_ = Point3D(np.array(points3d).T)
        points2D = points2d_.T.astype(np.float32)
        points3D = points3d_.T.astype(np.float32)

        try:
            _, K, kc, r, t = cv2.calibrateCamera(
                [points3D],
                [points2D],
                (width, height),
                None,
                None,
                None,
                None,
                cv2.CALIB_FIX_K1
                + cv2.CALIB_FIX_K2
                + cv2.CALIB_FIX_K3
                + cv2.CALIB_FIX_K4
                + cv2.CALIB_FIX_K5
                + cv2.CALIB_FIX_ASPECT_RATIO
                + cv2.CALIB_ZERO_TANGENT_DIST,
            )
        except cv2.error as err:
            print(err)
            return calib

        T = t[0]
        R = cv2.Rodrigues(r[0])[0]

        try:
            calib = Calib(width=width, height=height, T=T, R=R, K=K, kc=kc)
        except np.linalg.LinAlgError:
            print("no")
            pass
    # knowing that there's no distortion
    calib = calib.update(kc=None)

    return calib
