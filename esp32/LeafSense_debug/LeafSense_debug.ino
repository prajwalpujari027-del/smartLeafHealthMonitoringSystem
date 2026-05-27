#include <Wire.h>
#include <Adafruit_MLX90614.h>

// MLX90614 sensor on I2C (SDA=21, SCL=22)
Adafruit_MLX90614 mlx = Adafruit_MLX90614();

unsigned long lastReadTime = 0;
const unsigned long READ_INTERVAL = 1000; // Read every 1 second

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println();
  Serial.println("LeafSense Serial-Only Mode (No WiFi)");
  Serial.println("Reading MLX90614 from I2C and sending to COM7");
  Serial.println();

  Wire.begin(21, 22);

  if (!mlx.begin()) {
    Serial.println("[ERROR] MLX90614 sensor not detected. Check wiring:");
    Serial.println("  SDA -> GPIO 21");
    Serial.println("  SCL -> GPIO 22");
    Serial.println("  VIN -> 3.3V");
    Serial.println("  GND -> GND");
    while (1) {
      delay(1000);
    }
  }

  Serial.println("[OK] MLX90614 sensor detected and ready");
  Serial.println();
}

void loop() {
  unsigned long now = millis();

  if (now - lastReadTime >= READ_INTERVAL) {
    lastReadTime = now;

    float ambientTemp = mlx.readAmbientTempC();
    float objectTemp = mlx.readObjectTempC();

    // Format: "Ambient: XX.X C | Object: YY.Y C"
    // This format matches Flask's serial parser
    Serial.print("Ambient: ");
    Serial.print(ambientTemp, 1);
    Serial.print(" C | Object: ");
    Serial.print(objectTemp, 1);
    Serial.println(" C");
  }
}