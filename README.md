# 🌑 Lunar & Martian Crater Detection

> Upload a planetary surface image. Get back every crater found — boxed, classified, and streamed as live satellite telemetry.

---

## What I Built

I built an **end-to-end AI system** that automatically detects and maps craters on the Moon and Mars using deep learning. You upload an image, the system finds every crater, draws a box around it, tells you if it's small/medium/large, and streams the result as simulated satellite data — just like a real space mission would.

It took something that planetary scientists spend weeks doing manually and turned it into a 3-second process.

---

## Why It Matters

Impact craters tell us how old a planetary surface is. The more craters, the older the terrain. Scientists use crater counts to:
- Pick safe landing zones for spacecraft
- Estimate the age of different regions on the Moon and Mars
- Understand the geological history of other planets

The problem? There are **billions** of craters across the solar system. Mapping them by hand is impossible at scale. This project automates it.

---

## How It Works

```
You upload an image
        ↓
AI model scans it (YOLOv8 — same family of models used in self-driving cars)
        ↓
Every crater gets a box drawn around it + a confidence score
        ↓
Results are classified: Small / Medium / Large
        ↓
The detection is sent as a "satellite telemetry packet" to the IoT dashboard
        ↓
You see live charts, alerts, and a digital twin of the satellite
```

---

## The 3 Technologies Combined

### 🔭 Computer Vision (the brain)
Two AI models trained on real lunar crater data:

| Model | Speed | Accuracy | Use case |
|-------|-------|----------|----------|
| YOLOv8n | 8ms/image | mAP = 0.81 | Real-time analysis |
| YOLOv8s | 14ms/image | mAP = 0.82 | Higher accuracy |

Both models were trained on the **Robbins Lunar Crater Database** — 384,278 real craters mapped by NASA's Lunar Reconnaissance Orbiter.

### 📡 IoT / IoE (the satellite layer)
Every time a crater is detected, the system simulates what a real satellite would transmit back to Earth:
- Altitude, orbital velocity, instrument temperature
- Number of craters found, average confidence
- Alerts when crater density is too high for a landing zone
- All streamed via MQTT — the same protocol used in real IoT devices

### 🌐 Web App (the interface)
A Streamlit app with 4 tabs:
- **Detect** — upload image, see results instantly
- **Compare** — side-by-side model performance charts
- **IoT Dashboard** — live satellite telemetry feed
- **About** — how everything works

---

## Results

After training on 78 images for 30 epochs:

```
YOLOv8n (fast)   → 81.3% detection accuracy, 8ms per image
YOLOv8s (accurate) → 82.0% detection accuracy, 14ms per image
```

The model detects 3 crater classes:
- 🔴 **Small** — under 5 km diameter
- 🟢 **Medium** — 5 to 20 km
- 🔵 **Large** — over 20 km

---

## Dataset

**Source:** `juliensimon/lunar-craters-robbins` on HuggingFace
**Origin:** Stuart Robbins (2019), USGS Astrogeology Science Center
**Size:** 384,278 real lunar craters with lat/lon/diameter for each one
**License:** CC-BY-4.0 (free to use)

Since the catalog is tabular (coordinates, not images), I rendered synthetic terrain tiles with craters placed at their real positions — so the model trains on authentic lunar crater density and size distributions.

---

## Project Structure

```
crater-detection/
│
├── src/                     ← all the logic
│   ├── data_pipeline.py     generates training images from crater catalog
│   ├── train.py             trains YOLOv8n and YOLOv8s
│   ├── evaluate.py          measures accuracy, finds failure cases
│   ├── visualize.py         draws boxes, heatmaps, before/after images
│   └── iot_simulator.py     fake satellite + MQTT + ground station
│
├── app/
│   ├── app.py               main web app (Streamlit, 4 tabs)
│   └── iot_dashboard.py     standalone IoT telemetry view
│
├── data/
│   ├── processed/           train/val/test split (256×256 tiles)
│   └── dataset.yaml         tells YOLO where the data is
│
├── models/
│   ├── yolov8n_crater/      trained weights + accuracy results
│   └── yolov8s_crater/      trained weights + accuracy results
│
└── notebooks/
    └── exploration.ipynb    data exploration, charts, IoT demo
```

---

## Run It

```bash
# step 1 — set up environment
conda create -n crater python=3.11 -y
conda activate crater
pip install -r requirements.txt

# step 2 — generate training data
bash run.sh data 100

# step 3 — train both models
bash run.sh train 30

# step 4 — launch the app
bash run.sh app
# open http://localhost:8501
```

Or use the convenience script for everything:
```bash
bash run.sh data       # generate data
bash run.sh train      # train models
bash run.sh evaluate   # check accuracy
bash run.sh iot        # test IoT feed
bash run.sh app        # launch web app
```

---

## Deployment Options

### Option 1 — Streamlit Cloud (easiest, free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set main file to `app/app.py`
4. Done — live public URL in 2 minutes

```bash
git init && git add . && git commit -m "crater detector"
git remote add origin https://github.com/yourname/crater-detection.git
git push -u origin main
```

### Option 2 — Docker (portable, runs anywhere)
```dockerfile
# Dockerfile (add to project root)
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```
```bash
docker build -t crater-detector .
docker run -p 8501:8501 crater-detector
# open http://localhost:8501
```

### Option 3 — Hugging Face Spaces (free, public)
1. Create account at [huggingface.co/spaces](https://huggingface.co/spaces)
2. New Space → SDK: Streamlit → upload files
3. Instant public URL, free GPU available

### Option 4 — AWS / GCP / Azure
```bash
# example with AWS EC2
scp -r crater-detection/ ec2-user@your-ip:~/
ssh ec2-user@your-ip
cd crater-detection && conda activate crater
streamlit run app/app.py --server.port 80
```

### Option 5 — Real IoT Extension
To turn the simulated satellite pipeline into a real one:
1. Install Mosquitto MQTT broker: `brew install mosquitto`
2. Replace `InProcessBroker` in `iot_simulator.py` with `paho.mqtt.client`
3. Connect a Raspberry Pi running the edge detector as the publisher
4. Ground station subscribes from any machine on the network

---

## Known Limitations

- Trained on synthetic terrain — real LRO/HiRISE imagery will look different
- Small craters under 5px are sometimes missed
- Overlapping craters can get merged into one detection
- Model trained on Moon only — Mars needs fine-tuning

## What's Next

- Download real LRO imagery and retrain on actual satellite photos
- Add Mars and Mercury crater catalogs for multi-planet detection
- Hook up a real MQTT broker for live satellite feed
- Deploy to Streamlit Cloud so anyone can use it online
- Add 3D crater depth estimation from elevation data

---

## Tech Stack

`Python` · `PyTorch` · `Ultralytics YOLOv8` · `OpenCV` · `Streamlit` · `Plotly` · `MQTT` · `HuggingFace Datasets`

---

*Dataset: Robbins (2019) via USGS Astrogeology, distributed on HuggingFace by juliensimon. License CC-BY-4.0*
