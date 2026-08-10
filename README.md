
# Driver Drowsiness Detection System

A real-time driver fatigue monitoring system built with **transfer learning (MobileNetV2)**, trained on 15,000+ labeled driver behavior images across three states — **alert, sleepy, yawning** — and integrated with an **IoT-based alert mechanism** for real-time warnings. This demonstrates a full AI-to-hardware pipeline: camera → model inference → hardware alert.

## Overview

Driver fatigue is a leading cause of road accidents. This project detects early signs of drowsiness from a live camera feed and triggers a physical alert (buzzer/LED via Arduino) before the driver loses focus.

**Pipeline:**
```
Camera Feed → Face/Frame Preprocessing → MobileNetV2 (Transfer Learning) → Classification
    (alert / sleepy / yawning) → Confidence Thresholding → Serial Signal → Arduino Alert
```

## Features

- **Transfer learning** on MobileNetV2 (ImageNet weights), fine-tuned for 3-class driver state classification
- **Data augmentation** (rotation, brightness, zoom, horizontal flip) to improve generalization across lighting/angles
- **Real-time inference** from webcam with OpenCV
- **IoT alert integration** — sends a serial signal to an Arduino to trigger a buzzer/LED when drowsiness is detected for a sustained duration
- Modular, well-documented codebase for easy extension (e.g., adding eye-aspect-ratio or yawn-frequency heuristics)

## Repository Structure

```
driver-drowsiness-detection/
├── README.md
├── requirements.txt
├── data_preprocessing.py   # Load, clean, split dataset
├── data_augmentation.py    # Augmentation pipeline
├── model.py                # MobileNetV2 transfer learning model
├── train.py                # Training loop + checkpointing
├── evaluate.py              # Evaluation metrics + confusion matrix
├── inference.py             # Real-time webcam inference
├── iot_alert.py             # Serial communication with Arduino
├── drowsiness_alert.ino    # Arduino sketch for buzzer/LED alert
├── exploration.md          # Notes for EDA/experimentation (convert to .ipynb locally)
├── models/                  # Saved model weights (.h5) - not tracked in git
└── data/                    # Dataset (alert/ sleepy/ yawning/) - not tracked in git
```

## Dataset

Expected structure under `data/`:
```
data/
├── train/
│   ├── alert/
│   ├── sleepy/
│   └── yawning/
├── val/
│   ├── alert/
│   ├── sleepy/
│   └── yawning/
└── test/
    ├── alert/
    ├── sleepy/
    └── yawning/
```
Any labeled driver-behavior dataset works (e.g., the [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd) or a custom-collected set of 15,000+ images).

## Setup

```bash
git clone https://github.com/Manasvi27-04/driver-drowsiness-detection.git
cd driver-drowsiness-detection
pip install -r requirements.txt
```

## Usage

**1. Preprocess and split data**
```bash
python data_preprocessing.py --data_dir data/raw --output_dir data
```

**2. Train the model**
```bash
python train.py --data_dir data --epochs 20 --batch_size 32
```

**3. Evaluate**
```bash
python evaluate.py --model_path models/drowsiness_mobilenetv2.h5 --test_dir data/test
```

**4. Run real-time inference with IoT alert**
```bash
python inference.py --model_path models/drowsiness_mobilenetv2.h5 --serial_port COM3
```

## Model Details

| Component | Detail |
|---|---|
| Base model | MobileNetV2 (ImageNet pretrained, frozen base initially) |
| Input size | 224 × 224 × 3 |
| Classes | alert, sleepy, yawning |
| Fine-tuning | Top layers unfrozen after initial convergence, low LR fine-tune pass |
| Augmentation | Rotation, zoom, brightness, horizontal flip |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |

## IoT Alert Integration

`iot_alert.py` opens a serial connection to an Arduino running `drowsiness_alert.ino`. When the model predicts **sleepy** or **yawning** with high confidence for a sustained number of consecutive frames, it sends a signal (`'1'`) over serial, which triggers a buzzer and LED on the Arduino. This closes the loop from AI inference to physical hardware response.

## Results

_Add your trained model's accuracy/precision/recall/F1 and confusion matrix here once training is complete._

## Future Work

- Incorporate eye-aspect-ratio (EAR) and mouth-aspect-ratio (MAR) as auxiliary signals
- Extend to multimodal fusion (audio cues, physiological signals) — see companion project [DriveSafeAI 2.0]
- Deploy quantized model (TFLite) for edge devices (Raspberry Pi / Jetson Nano)

## License

MIT
