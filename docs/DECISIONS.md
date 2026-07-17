# Decisions

Binding conventions. Every notebook and module conforms. Changing anything
here requires updating every place that states it, in the same commit.

## Coordinates

- `+x` right, `+y` down, `+z` forward. Same convention for world and camera
  (OpenCV / COLMAP). Right-handed: right cross down = forward.
- World up is `(0, -1, 0)`; it is the default `up` in `look_at`.
- `look_at` returns the world-to-camera rotation with rows
  `(right, down, forward)` as world vectors.
- Camera model: `u = fx x/z + cx`, `v = fy y/z + cy`. Principal point at the
  image center. Kernels are evaluated at integer pixel coordinates; means
  stay float. No distortion anywhere in this project.

## Gaussians

- Kernel is unnormalized: `exp(-0.5 d^T Sigma^-1 d)`, peak exactly 1.
  Opacity is a separate scalar meaning "opacity at the center".
- Covariance is always built as a product, never stored entrywise:
  2D `Sigma = R S S^T R^T`, 3D the same, or `B diag(s^2) B^T` for a direct
  basis. Positive semi-definite by construction.
- Quaternions are `(w, x, y, z)`, w first, matching 3DGS ply field order
  `rot_0..rot_3`. Convert via normalize-then-rotation-matrix, so any
  4-vector maps to a valid rotation.

## Projection (EWA)

- `Sigma_cam = W Sigma W^T` with `W = cam.R` (exact, affine).
- `J = [[fx/z, 0, -fx x/z^2], [0, fy/z, -fy y/z^2]]` at the camera-space
  mean, shape 2x3. The 2x3 shape performs the depth marginalization.
- `Sigma_2d = J W Sigma W^T J^T`, computed as `T = J @ W`, `T Sigma T^T`.
- Deliberate simplifications relative to the official implementation, each
  with its scheduled fix: no `1.3 tan(fov/2)` clamp on `x/z, y/z` inside `J`
  (added in phase D), no opacity compensation for dilation (added when the
  artifact appears on real scenes, see Mip-Splatting).

## Renderer constants

```
DILATION  = 0.3     screen-space low-pass: Sigma_2d += 0.3 I
ALPHA_MIN = 1/255   skip contributions that cannot change an 8-bit pixel
ALPHA_MAX = 0.99    keep (1 - alpha) away from zero for the backward pass
T_STOP    = 1e-4    region transmittance below this: remaining splats skipped
z_near    = 0.05    cull before the Jacobian sees a small z
```

- Conic = inverse 2D covariance, computed once per splat with the hand
  2x2 inverse. Per-pixel exponent:
  `power = -0.5 (A dx^2 + C dy^2) - B dx dy`, contribute only if
  `power <= 0` territory yields `alpha >= ALPHA_MIN`.
- Bounding radius: `mid = (a + c)/2`,
  `lam_max = mid + sqrt(max(0.1, mid^2 - det))`,
  `radius = ceil(3 sqrt(lam_max))`, using the dilated covariance.
- Sort: one global order per frame by camera-space depth of the mean, near
  to far, front-to-back compositing. Popping under order flips is a known,
  accepted artifact; do not "fix" it without a phase E discussion.
- Background composited last: `img += T_final * bg`.

## Spherical harmonics color (notebook 06+)

- Coefficient layout `(N, 16, 3)`: 16 basis functions times RGB, degree
  bands `l = 0..3`, official 3DGS order and sign convention. Constants
  `C0..C3` live in `gsplat_edu/sh.py`; trained plys decode only under
  exactly these signs.
- Decode: `color = clip(sum_i b_i(d) sh_i + 0.5, 0, None)`. The 0.5 offset
  makes the all-zero coefficient vector middle gray; the clip only floors.
- View direction `d = normalize(mean - cam.eye)`, one per splat per frame
  (the official preprocess approximation).
- The renderer stays color-agnostic: callers run `eval_sh` and pass RGB to
  `render_gaussians`.

## Parameter activations (training and ply loading)

```
opacity = sigmoid(raw)
scale   = exp(raw)          (store log-scale)
rotation: normalize(quat)
```

## Project mechanics

- Package name `gsplat_edu`, src layout, editable install.
- Notebook sources in `notebooks_src/` (jupytext percent), executed output
  in `notebooks/`, built only via `tools/build_notebooks.py`.
- Toy scene of record: `sphere_shell` (tangent disks on the unit sphere,
  `s_t = 0.06`, `s_n = 0.01`, opacity 0.8, color = position remapped).
- Reference camera of record: eye `(0, 0, -3)`, target origin,
  `fx = fy = 600`, 512x512.
