## infra-vis

This repository generates simple infrastructure flow visualizations and MP4 animations from JSON process definitions.

What this does
- Provide a data-driven way to describe nodes and flows in JSON (`inputs/*.json`).
- Render static previews (HTML + PNG) and animated MP4s using Plotly (frames) + OpenCV (encoding).

Quick start

1. Create a Python 3.12 virtual environment and activate it:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Run the visualizer with an input JSON (example provided):

```powershell
.venv\Scripts\python.exe scripts\animate_flow_opencv.py inputs/example_azure_flow.json
```

Outputs
- Files are written to the `media/` folder:
    - `*.html` — interactive Plotly preview
    - `*.png` — static snapshot
    - `*.mp4` — encoded animation (per-frame rendering + OpenCV)

Notes & tips
- The script auto-calculates animation length based on the route length (number of segments). You can adjust `speed` on a `route` flow to make the moving marker faster or slower.
- Lines are rendered as muted dotted gray by default; node colors and flow colors are taken from the JSON but can be overridden.
- Frame rendering is parallelized (ThreadPoolExecutor) — you can tune `max_workers` in `create_animated_mp4` to match your CPU.
- If OpenCV (`cv2`) or Plotly image export (`kaleido`) are not installed, only HTML will be produced.

Files to look at
- `scripts/animate_flow_opencv.py` — main visualizer (renders static + animated outputs).
- `inputs/` — example JSON inputs and schema.
- `media/` — generated outputs (HTML/PNG/MP4).

If you'd like, I can add a short CLI wrapper, or export GIFs instead of MP4s.

