# Machine Hallucinations

A full-stack application that explores generative AI models, specifically featuring a latent space walk implementation. It includes a Python backend for data ingestion and generative computations, alongside a lightweight web-based frontend for interactive visualization.

## Project Structure

```text
machine-hallucinations-main/
├── backend/
│   ├── __init__.py
│   ├── config.py         # Configuration settings for the backend and models
│   ├── ingest.py         # Data ingestion and preprocessing scripts
│   ├── latent_walk.py    # Core logic for interpolating and exploring latent spaces
│   └── main.py           # Main application entry point (e.g., FastAPI/Flask server)
├── frontend/
│   ├── app.js            # Frontend logic and API integration
│   ├── index.html        # Main HTML interface
│   └── styles.css        # UI styling
├── .gitignore            # Ignored files (e.g., venv, __pycache__)
├── requirements.txt      # Python backend dependencies
└── readme.md             # Project documentation
```

## Features
- **Latent Walk Generation:** Traverse the latent space of a generative model to create smooth transitions between generated outputs.
- **Data Ingestion Pipeline:** Modular data loading and processing through `ingest.py`.
- **Interactive Web UI:** A decoupled JavaScript/HTML/CSS frontend to control parameters and view the "hallucinations" in real-time.

## Prerequisites
- Python 3.8+
- Modern Web Browser

## Technology Stack
- **Backend:** Python with FastAPI (for high performance and asynchronous handling).
- **AI/Generative Engine:** PyTorch with Hugging Face `diffusers` (or a lightweight StyleGAN implementation) to generate latent space interpolations between the satellite images.
- **Frontend:** HTML5, CSS3, and JavaScript utilizing Three.js or WebGL. The frontend must handle the fluid, particle-based rendering of the images.

## Core Mechanics & Features
- **Data Ingestion:** A script or endpoint to load a local folder of satellite images and preprocess them (resize, crop, normalize).
- **The "Hallucination" (Backend):** Write a Python script that takes these images, projects them into a latent space, and generates a continuous "latent walk" (a seamless, morphing video loop or a stream of transitional frames).


## Installation & Setup

### 1. Backend Setup
Navigate to the project directory and set up a virtual environment:
```bash
cd machine-hallucinations-main
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
```

Install the required Python dependencies:
```bash
pip install -r requirements.txt
```

### 2. Frontend Setup
The frontend consists of static files. You can serve them using any basic HTTP server. For example:
```bash
cd frontend
python -m http.server 8080
```

## Usage

1. **Start the Backend Server:**
   ```bash
   cd backend
   python main.py
   ```
   *(Ensure the backend is running and listening for API requests on the configured port in `config.py`)*

2. **Launch the UI:**
   Open your web browser and navigate to `http://localhost:8080` (or the port you chose for your frontend server) to interact with the latent walk visualizations.

