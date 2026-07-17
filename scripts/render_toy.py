"""Render the Phase A test scene to out/toy_render.png."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gsplat_edu import Camera, render_gaussians, sphere_shell

means, covs, colors, opac = sphere_shell()
cam = Camera.looking_at(eye=[0, 0, -3], target=[0, 0, 0], fx=600, fy=600, H=512, W=512)
img, _ = render_gaussians(cam, means, covs, colors, opac)

os.makedirs("out", exist_ok=True)
plt.imsave("out/toy_render.png", np.clip(img, 0, 1))
print("wrote out/toy_render.png")
