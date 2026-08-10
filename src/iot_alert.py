"""
iot_alert.py

Handles serial communication with an Arduino running
arduino/drowsiness_alert.ino. Sends '1' to trigger the buzzer/LED alert
and '0' to clear it.
"""

import time

try:
    import serial
except ImportError:
    serial = None


class IoTAlertSystem:
    def __init__(self, port: str = None, baud_rate: int = 9600, timeout: int = 1):
        self.enabled = port is not None and serial is not None
        self.connection = None

        if self.enabled:
            try:
                self.connection = serial.Serial(port, baud_rate, timeout=timeout)
                time.sleep(2)  # allow Arduino to reset after connection
                print(f"[IoTAlertSystem] Connected to {port}")
            except Exception as e:
                print(f"[IoTAlertSystem] Could not connect to {port}: {e}")
                self.enabled = False
        else:
            print("[IoTAlertSystem] Running without hardware alert (no serial port configured).")

    def trigger_alert(self):
        if self.enabled and self.connection:
            self.connection.write(b"1")
        else:
            print("[IoTAlertSystem] ALERT: Drowsiness detected! (simulated, no hardware connected)")

    def clear_alert(self):
        if self.enabled and self.connection:
            self.connection.write(b"0")

    def close(self):
        if self.enabled and self.connection:
            self.connection.close()
