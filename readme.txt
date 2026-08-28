Step 0 — Environment

cd machine-hallucinations
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install PyTorch FIRST, matching your hardware:
pip install torch --index-url https://download.pytorch.org/whl/cu121   # NVIDIA (CUDA 12.1)
# pip install torch                                                    # CPU / Apple Silicon

pip install -r requirements.txt
mkdir -p data/raw          # backend subfolders auto-create on first run

Step 1 — Initialize the dataset
Drop 5–100 satellite images (.jpg/.png/.tif/.webp) into data/raw/. Good sources: UC Merced Land Use (2,100 labeled 256×256 tiles — ideal), Sentinel-2 RGB composites, or NASA Visible Earth crops. Then normalize:

bash

python -m backend.ingest                    # → data/processed/*.png  (512×512)

Step 2 — Render the hallucination (latent walk)
Option A — CLI (recommended first run):

bash

# Fast smoke test (works on CPU, ~2–5 min):
python -m backend.latent_walk --steps 16 --max-keyframes 4

# Exhibition quality (GPU): 24 anchors × 48 frames = 1,152 frames (~38 s loop)
python -m backend.latent_walk --steps 48 --fps 30 --dwell 0.18 --noise 0.035

Option B — via the API (or just open the site and press the GENERATE button):

curl -X POST http://localhost:9000/api/generate \
     -H "Content-Type: application/json" \
     -d '{"steps_per_transition": 48, "fps": 30, "noise_level": 0.035}'
curl http://localhost:9000/api/jobs/<job_id>     # poll progress

First render downloads the VAE (~335 MB, cached in ~/.cache/huggingface).

Step 3 — Start the web server

uvicorn backend.main:app --host 0.0.0.0 --port 9000
# → open http://localhost:8000

Controls & tuning
Key	Effect	Backend knob	Effect
1/2/3	particle density (9k → 102k)	--dwell	how long each image crystallizes before dissolving
B	bloom on/off	--noise	mid-morph hallucination intensity (0 = clean morph)
+ / −	flow turbulence	--easing	travel rhythm (smootherstep ≈ Anadol's cadence)
SPACE	pause	--max-keyframes	cap anchor count (farthest-point sampled)
G	render a fresh walk	MH_IMAGE_SIZE=768	render at higher resolution (env var)
move pointer	repulsion vortex	
