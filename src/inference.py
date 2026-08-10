"""
inference.py

Real-time webcam inference: captures frames, classifies driver state
(alert / sleepy / yawning), and triggers an IoT hardware alert when
drowsiness is detected for a sustained number of consecutive frames.
"""

import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from iot_alert import IoTAlertSystem

CLASS_NAMES = ["alert", "sleepy", "yawning"]
IMG_SIZE = (224, 224)
DROWSY_CLASSES = {"sleepy", "yawning"}
CONSECUTIVE_FRAME_THRESHOLD = 15  # ~0.5s at 30fps, tune to your camera's fps
CONFIDENCE_THRESHOLD = 0.6


def preprocess_frame(frame):
    resized = cv2.resize(frame, IMG_SIZE)
    normalized = resized.astype("float32") / 255.0
    return np.expand_dims(normalized, axis=0)


def main(model_path, serial_port, camera_index=0):
    model = load_model(model_path)
    alert_system = IoTAlertSystem(port=serial_port)
    cap = cv2.VideoCapture(camera_index)

    drowsy_frame_count = 0
    alert_active = False

    print("Starting real-time inference. Press 'q' to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame from camera.")
                break

            input_tensor = preprocess_frame(frame)
            preds = model.predict(input_tensor, verbose=0)[0]
            class_idx = int(np.argmax(preds))
            confidence = float(preds[class_idx])
            label = CLASS_NAMES[class_idx]

            if label in DROWSY_CLASSES and confidence >= CONFIDENCE_THRESHOLD:
                drowsy_frame_count += 1
            else:
                drowsy_frame_count = 0

            if drowsy_frame_count >= CONSECUTIVE_FRAME_THRESHOLD and not alert_active:
                alert_system.trigger_alert()
                alert_active = True
            elif drowsy_frame_count == 0 and alert_active:
                alert_system.clear_alert()
                alert_active = False

            display_text = f"{label} ({confidence:.2f})"
            color = (0, 0, 255) if label in DROWSY_CLASSES else (0, 255, 0)
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Driver Drowsiness Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        alert_system.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time driver drowsiness inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--serial_port", default=None, help="e.g. COM3 or /dev/ttyUSB0; omit to run without hardware")
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()
    main(args.model_path, args.serial_port, args.camera_index)
