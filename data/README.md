# Data

Nothing in this repo downloads data. Place scenes here yourself:

```
data/<scene>/point_cloud.ply     a trained 3DGS point cloud (notebook 07+)
```

Sources: the official 3DGS pretrained models from the Inria release
(graphdeco-inria/gaussian-splatting readme links them), or the
`point_cloud/iteration_30000/point_cloud.ply` from any training run of the
official code or gsplat. Everything in this directory except this file is
gitignored.

`data/_synthetic/` is written by notebook 07: the toy sphere dressed in the
trained-ply format, used as a stand-in whenever no real scene is present.
Notebook 07 prefers any real scene it finds here; drop one in and rebuild.
