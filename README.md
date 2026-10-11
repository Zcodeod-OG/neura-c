# Neura-C: outline and motion encoding for low-resolution electrotactile vision

Code for the preprint *Outline and Motion Encoding of RGB-D Scenes for Low-Resolution Electrotactile
Vision: A Simulation Study* (Amarsh N Choudhary, IIT Delhi, 2026).

The simulator turns RGB-D video into a sparse line drawing (depth outlines weighted by nearness, plus speed
streaks behind moving objects), removes the wearer's own camera motion using depth, and renders the result on a
model electrode grid (16x16 to 32x32, four intensity levels, optional temporal hysteresis).

![Synthetic scene](results/still.png)
![Real Kinect sequence](results/lounge_still.png)

## Files

| File | What it does |
| --- | --- |
| `scene.py` | Synthetic RGB-D scene with ground-truth labels (wall with posters, box, walking person, fast ball) |
| `encoder.py` | Depth outlines, optical-flow speed streaks, depth-hole filling, ego-motion removal (homography or depth + PnP) |
| `grid.py` | Electrode grid model, raw-image baseline, ground-truth boundaries, `GridSmoother` hysteresis |
| `run.py` | Synthetic experiment: boundary F1 and electrodes lit per grid size |
| `run_real.py` | Real RGB-D experiment: electrodes lit, flicker, spurious streaks |
| `results/` | Figures, demo GIFs and metrics reported in the paper |

## Run

```
pip install -r requirements.txt
python3 run.py                                  # synthetic scene
python3 run_real.py <sequence_dir> lounge       # folder with color/ and depth/ PNGs, depth in mm
```

The real sequence in the paper is frames 1399 to 1639 of the Open3D Lounge RGB-D sequence
([download](https://github.com/isl-org/open3d_downloads/releases/download/20220301-data/LoungeRGBDImages.zip)).

## Limitations

These are signal-level simulation results, not perceptual results. Nothing here drives stimulation hardware.

## License

MIT
