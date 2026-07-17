"""Pinhole camera. Convention (OpenCV/COLMAP): +x right, +y down, +z forward.

This project uses the same y-down convention for the world, so "world up"
is (0, -1, 0). One convention everywhere; real COLMAP data arrives in it.
"""
import numpy as np
from dataclasses import dataclass


def look_at(eye, target, up=(0, -1, 0)):
    """World-to-camera rotation. Rows are (right, down, forward) as world vectors."""
    eye, target, up = (np.asarray(a, float) for a in (eye, target, up))
    f = target - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, up)
    r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f])


@dataclass
class Camera:
    R: np.ndarray      # (3,3) world-to-camera rotation
    eye: np.ndarray    # (3,) camera center, world coords
    fx: float
    fy: float
    cx: float
    cy: float
    H: int
    W: int

    @classmethod
    def looking_at(cls, eye, target, fx, fy, H, W, up=(0, -1, 0)):
        return cls(look_at(eye, target, up), np.asarray(eye, float),
                   fx, fy, W / 2, H / 2, H, W)

    def world_to_cam(self, P):
        return (np.asarray(P, float) - self.eye) @ self.R.T

    def project(self, Pc):
        """Camera coords -> pixel coords. Pc is (..., 3), z must be > 0."""
        z = Pc[..., 2]
        u = self.fx * Pc[..., 0] / z + self.cx
        v = self.fy * Pc[..., 1] / z + self.cy
        return u, v, z
