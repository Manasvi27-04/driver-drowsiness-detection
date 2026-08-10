/*
  drowsiness_alert.ino

  Listens on serial for '1' (trigger alert) or '0' (clear alert) sent
  from the Python inference pipeline (see src/iot_alert.py) and drives
  a buzzer + LED accordingly.
*/

const int BUZZER_PIN = 8;
const int LED_PIN = 13;

void setup() {
  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(LED_PIN, LOW);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == '1') {
      digitalWrite(BUZZER_PIN, HIGH);
      digitalWrite(LED_PIN, HIGH);
    } else if (command == '0') {
      digitalWrite(BUZZER_PIN, LOW);
      digitalWrite(LED_PIN, LOW);
    }
  }
}
