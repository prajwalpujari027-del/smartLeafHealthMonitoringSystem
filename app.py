from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model, model_from_json
from werkzeug.utils import secure_filename
import numpy as np
import cv2
import json
import os
import threading
from datetime import datetime, timezone
from uuid import uuid4
import socket
import time
import re

try:
    from pyngrok import ngrok
except Exception:
    ngrok = None

try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None

app = Flask(__name__)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MODEL_PATH = "model/plant_disease_model.keras"
CLASS_NAMES_PATH = "model/class_names.json"
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM8")
SERIAL_BAUD = int(os.getenv("SERIAL_BAUD", "115200"))
SERIAL_ENABLED = os.getenv("SERIAL_ENABLED", "1") == "1"
CONF_THRESHOLD = 0.70
USE_NGROK = os.getenv("USE_NGROK", "1") == "1"
NGROK_PORT = int(os.getenv("NGROK_PORT", "5000"))
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN", "")

# Load model and classes once at startup
model = None
class_names = []

try:
    model = load_model(MODEL_PATH)
    print(f"[MODEL] Successfully loaded model from {MODEL_PATH}")
    try:
        import io
        buf = io.StringIO()
        model.summary(print_fn=lambda s: buf.write(s + "\n"))
        print("[MODEL] Summary:\n" + buf.getvalue())
    except Exception as exc_s:
        print(f"[MODEL] Could not print model summary: {exc_s}")

    # If metadata.json exists inside the model folder, print it for verification
    try:
        meta_path = os.path.join(MODEL_PATH, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta_text = mf.read()
            print(f"[MODEL] metadata.json: {meta_text}")
    except Exception as exc_m:
        print(f"[MODEL] Could not read metadata.json: {exc_m}")
except Exception as exc:
    print(f"[ERROR] Failed to load model from {MODEL_PATH}: {exc}")
    import traceback
    traceback.print_exc()

try:
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    print(f"[MODEL] Loaded {len(class_names)} classes: {class_names[:3]}...")
except Exception as exc:
    print(f"[ERROR] Failed to load class names from {CLASS_NAMES_PATH}: {exc}")

sensor_lock = threading.Lock()
sensor_data = {
    "ambient": None,
    "object": None,
    "updated": None,
}


def get_local_ip():
    """Return the machine's local LAN IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # This doesn't need to be reachable; it's only used to determine the outbound interface.
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


def start_ngrok(port):
    """Start ngrok tunnel and return the public URL."""
    if ngrok is None:
        print("[NGROK] pyngrok is not installed. Install with: pip install pyngrok")
        return None

    if NGROK_AUTH_TOKEN:
        try:
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
        except Exception as exc:
            print(f"[NGROK] Could not set auth token: {exc}")
            return None
    else:
        print("[NGROK] No NGROK_AUTH_TOKEN provided. Set NGROK_AUTH_TOKEN or run 'ngrok authtoken <token>'.")
        return None

    try:
        tunnel = ngrok.connect(port, "http")
        public_url = tunnel.public_url
        print(f"[NGROK] Public tunnel URL: {public_url}")
        return public_url
    except Exception as exc:
        print(f"[NGROK] Could not start ngrok tunnel: {exc}")
        return None


def _update_sensor_values(ambient_val, object_val):
    """Update sensor data with thread safety"""
    with sensor_lock:
        sensor_data["ambient"] = round(float(ambient_val), 1)
        sensor_data["object"] = round(float(object_val), 1)
        sensor_data["updated"] = datetime.now(timezone.utc).isoformat()
        print(f"[SENSOR_DATA] Updated: ambient={sensor_data['ambient']}, object={sensor_data['object']}")


def _parse_serial_line(line):
    """Parse temperature data from serial line
    
    Supports:
    1. "Ambient: 29.12 C | Object: 31.84 C"
    2. "29.1,31.8"
    """
    line = line.strip()
    if not line:
        return None

    # Format 1: "Ambient: X.XX C | Object: Y.YY C"
    match = re.search(
        r"Ambient\s*:\s*(-?\d+(?:\.\d+)?)\s*C.*Object\s*:\s*(-?\d+(?:\.\d+)?)\s*C",
        line,
        flags=re.IGNORECASE,
    )
    if match:
        ambient = float(match.group(1))
        obj_temp = float(match.group(2))
        print(f"[PARSE] Format 1: ambient={ambient}, object={obj_temp}")
        return ambient, obj_temp

    # Format 2: "29.1,31.8" (CSV)
    match = re.search(r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", line)
    if match:
        ambient = float(match.group(1))
        obj_temp = float(match.group(2))
        print(f"[PARSE] Format 2 (CSV): ambient={ambient}, object={obj_temp}")
        return ambient, obj_temp

    # If nothing matched, log it for debugging
    print(f"[PARSE] Could not parse: '{line}'")
    return None


def list_serial_ports():
    """List available serial ports"""
    if serial is None:
        return []
    
    ports = []
    try:
        for port, desc, hwid in list_ports.comports():
            ports.append(f"{port} - {desc}")
    except Exception as e:
        print(f"[SERIAL] Could not list ports: {e}")
    
    return ports


def serial_reader_worker():
    """Read temperature data from COM port continuously"""
    if not SERIAL_ENABLED:
        print("[SERIAL] Disabled (SERIAL_ENABLED=0)")
        return
    
    if serial is None:
        print("[SERIAL] pyserial is not installed. Run: pip install pyserial")
        return

    print("[SERIAL] Serial reader thread started")
    print(f"[SERIAL] Looking for port: {SERIAL_PORT} @ {SERIAL_BAUD} baud")
    print("[SERIAL] Available ports:")
    for port in list_serial_ports():
        print(f"         {port}")

    while True:
        try:
            print(f"[SERIAL] Attempting to connect to {SERIAL_PORT}...")
            with serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=2) as ser:
                print(f"[SERIAL] ✅ Connected to {SERIAL_PORT} @ {SERIAL_BAUD} baud")
                time.sleep(2)  # Wait for Arduino/ESP32 to be ready
                ser.reset_input_buffer()

                # Read data continuously
                consecutive_errors = 0
                while True:
                    try:
                        raw = ser.readline().decode("utf-8", errors="ignore").strip()
                        
                        if not raw:
                            continue

                        # Try to parse the line
                        parsed = _parse_serial_line(raw)
                        if parsed is None:
                            consecutive_errors += 1
                            if consecutive_errors < 5:
                                print(f"[SERIAL] ⚠️  Unparseable line: '{raw}'")
                            continue

                        # Successfully parsed
                        consecutive_errors = 0
                        ambient_val, object_val = parsed
                        _update_sensor_values(ambient_val, object_val)

                    except Exception as e:
                        print(f"[SERIAL] Read error: {e}")
                        consecutive_errors += 1
                        if consecutive_errors > 10:
                            break
                        time.sleep(0.1)

        except serial.SerialException as exc:
            print(f"[SERIAL] ❌ Connection error: {exc}")
            print(f"[SERIAL] Retrying in 3 seconds...")
            time.sleep(3)
        except Exception as exc:
            print(f"[SERIAL] Unexpected error: {exc}")
            print(f"[SERIAL] Retrying in 5 seconds...")
            time.sleep(5)


# ===== Solutions Dictionary =====
solutions = {
    "Tomato___Leaf_Mold": {
        "advice": "Reduce greenhouse humidity and improve ventilation.",
        "fertilizer": "Apply magnesium fertilizer for leaf health.",
        "pesticide": "Use sulfur-based fungicide.",
        "organic": "Use diluted milk spray.",
        "product_link": "https://www.amazon.in/s?k=sulfur+fungicide+plants",
        "youtube": "https://www.youtube.com/results?search_query=leaf+mold+tomato+treatment",
    },
    "Tomato___Septoria_leaf_spot": {
        "advice": "Remove affected leaves and avoid water splash.",
        "fertilizer": "Use balanced fertilizer with micronutrients.",
        "pesticide": "Apply copper fungicide every 7 days.",
        "organic": "Use neem oil spray.",
        "product_link": "https://www.amazon.in/s?k=copper+fungicide+plants",
        "youtube": "https://www.youtube.com/results?search_query=septoria+leaf+spot+tomato",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "advice": "Increase humidity and wash leaves with water spray.",
        "fertilizer": "Use nitrogen-balanced fertilizer.",
        "pesticide": "Apply miticide or insecticidal soap.",
        "organic": "Use neem oil spray every 3 days.",
        "product_link": "https://www.amazon.in/s?k=miticide+neem+oil+plants",
        "youtube": "https://www.youtube.com/results?search_query=spider+mites+treatment+tomato",
    },
    "Tomato___Target_Spot": {
        "advice": "Avoid wet leaves and maintain proper spacing.",
        "fertilizer": "Use calcium nitrate fertilizer.",
        "pesticide": "Apply broad-spectrum fungicide.",
        "organic": "Use compost tea spray.",
        "product_link": "https://www.amazon.in/s?k=fungicide+for+tomato+plants",
        "youtube": "https://www.youtube.com/results?search_query=target+spot+tomato+treatment",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "advice": "Control whiteflies immediately and isolate infected plants.",
        "fertilizer": "Use micronutrient spray to reduce stress.",
        "pesticide": "Apply imidacloprid-based pesticide.",
        "organic": "Use yellow sticky traps.",
        "product_link": "https://www.amazon.in/s?k=imidacloprid+whitefly+control",
        "youtube": "https://www.youtube.com/results?search_query=tomato+yellow+leaf+curl+virus+treatment",
    },
    "Tomato___Tomato_mosaic_virus": {
        "advice": "Remove infected plants and disinfect tools.",
        "fertilizer": "Use organic compost to improve soil quality.",
        "pesticide": "No direct cure. Focus on prevention.",
        "organic": "Use disease-free seeds and crop rotation.",
        "product_link": "https://www.amazon.in/s?k=organic+compost+for+plants",
        "youtube": "https://www.youtube.com/results?search_query=tomato+mosaic+virus+control",
    },
    "Tomato___Early_blight": {
        "advice": "Remove lower infected leaves; stake plants for airflow.",
        "fertilizer": "Apply balanced NPK fertilizer to strengthen plant.",
        "pesticide": "Use mancozeb or chlorothalonil fungicide.",
        "organic": "Spray baking soda solution (1 tbsp per litre).",
        "product_link": "https://www.amazon.in/s?k=mancozeb+fungicide",
        "youtube": "https://www.youtube.com/results?search_query=tomato+early+blight+treatment",
    },
    "Tomato___Late_blight": {
        "advice": "Remove and destroy infected tissue immediately.",
        "fertilizer": "Reduce nitrogen; boost potassium.",
        "pesticide": "Apply metalaxyl or cymoxanil fungicide.",
        "organic": "Use copper-based organic spray.",
        "product_link": "https://www.amazon.in/s?k=metalaxyl+fungicide",
        "youtube": "https://www.youtube.com/results?search_query=tomato+late+blight+control",
    },
    "Tomato___Bacterial_spot": {
        "advice": "Avoid overhead irrigation; remove infected debris.",
        "fertilizer": "Use calcium-rich fertilizer.",
        "pesticide": "Apply copper bactericide weekly.",
        "organic": "Spray diluted hydrogen peroxide solution.",
        "product_link": "https://www.amazon.in/s?k=copper+bactericide+plants",
        "youtube": "https://www.youtube.com/results?search_query=tomato+bacterial+spot+treatment",
    },
    "Tomato___healthy": {
        "advice": "Plant looks healthy! Maintain current care routine.",
        "fertilizer": "Continue balanced NPK fertilizer schedule.",
        "pesticide": "No treatment needed.",
        "organic": "Consider compost tea for ongoing nutrition.",
        "product_link": "https://www.amazon.in/s?k=organic+tomato+fertilizer",
        "youtube": "https://www.youtube.com/results?search_query=tomato+plant+care+tips",
    },
    "Tomato__Random": {
        "advice": "⚠️ This is not a tomato plant image. Please upload a clear tomato leaf image.",
        "fertilizer": "N/A - Not a tomato plant",
        "pesticide": "N/A - Not a tomato plant",
        "organic": "N/A - Not a tomato plant",
        "product_link": "https://www.amazon.in/s?k=tomato+seeds",
        "youtube": "https://www.youtube.com/results?search_query=how+to+grow+tomato+plants",
    },
    "Unknown": {
        "advice": "Disease class not recognized.",
        "fertilizer": "Inspect plant manually.",
        "pesticide": "No recommendation.",
        "organic": "Monitor plant condition.",
        "product_link": "#",
        "youtube": "#",
    },
}


# ===== Routes =====

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/sensor", methods=["POST"])
def receive_sensor():
    """Receive sensor data via HTTP (if using HTTP method)"""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON"}), 400

    ambient = data.get("ambient")
    obj = data.get("object")

    if ambient is None or obj is None:
        return jsonify({"error": "Missing fields: ambient, object"}), 400

    try:
        ambient_val = float(ambient)
        object_val = float(obj)
    except (TypeError, ValueError):
        return jsonify({"error": "ambient and object must be numbers"}), 400

    _update_sensor_values(ambient_val, object_val)
    print(f"[HTTP] /sensor updated ambient={ambient_val:.1f}, object={object_val:.1f}")

    return jsonify({"status": "ok"})


@app.route("/update_sensor", methods=["POST"])
def update_sensor():
    """Receive sensor data from ESP32 or other HTTP clients"""
    return receive_sensor()


@app.route("/sensor", methods=["GET"])
def get_sensor():
    """Get current sensor data"""
    with sensor_lock:
        return jsonify({
            "ambient": sensor_data["ambient"],
            "object": sensor_data["object"],
        })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    filename = secure_filename(file.filename)
    filepath = os.path.join(
        UPLOAD_DIR,
        f"{uuid4().hex}_{filename}"
    )

    file.save(filepath)

    try:

        # =========================
        # CNN IMAGE PREDICTION
        # =========================

        img = cv2.imread(filepath)

        if img is None:
            return jsonify({"error":"Invalid image"}),400

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        img = cv2.resize(
            img,
            (150,150)
        )

        img = img/255.0

        img=np.expand_dims(
            img,
            axis=0
        )

        prediction=model.predict(
            img,
            verbose=0
        )

        class_index=int(
            np.argmax(prediction)
        )

        confidence=float(
            np.max(prediction)
        )

        if (
            0 <= class_index < len(class_names)
            and confidence >= CONF_THRESHOLD
        ):
            disease = class_names[class_index]
        else:
            disease = "Unknown"

        info=solutions.get(
            disease,
            {}
        )

        # =========================
        # TEMPERATURE DATA
        # =========================

        with sensor_lock:
            leaf_temp = sensor_data["object"]
            ambient_temp = sensor_data["ambient"]

        # =========================
        # TEMPERATURE ANALYSIS
        # =========================

        temp_warning = "Normal"
        temp_status = "Healthy"
        risk_level = "LOW"
        action_priority = "LOW"

        watering = (
            "Maintain standard watering schedule "
            "(check soil moisture before watering)"
        )

        irrigation = (
            "Maintain normal irrigation"
        )

        thermal_action = (
            "Plant thermal condition stable"
        )

        temp_score = 0

        diff = 0

        if (
            leaf_temp is not None
            and ambient_temp is not None
        ):

            diff = leaf_temp - ambient_temp

            # Extreme heat
            if leaf_temp > 38:
                temp_warning = "Extreme leaf temperature"
                temp_status = "Severe Heat Stress"
                risk_level = "HIGH"
                action_priority = "URGENT"
                temp_score = 30
                watering = (
                    "Increase watering frequency "
                    "(early morning + evening)"
                )
                irrigation = (
                    "Provide shade net or drip irrigation"
                )
                thermal_action = (
                    "Cool root zone immediately"
                )

            # High temperature
            elif leaf_temp > 34:
                temp_warning = (
                    "Leaf temperature elevated"
                )
                temp_status = (
                    "Heat Stress"
                )
                risk_level = "MEDIUM"
                action_priority = "MODERATE"
                temp_score = 18
                watering = (
                    "Increase watering slightly"
                )
                irrigation = (
                    "Water during morning only"
                )
                thermal_action = (
                    "Mulch soil to retain moisture"
                )

            # Cold condition
            elif leaf_temp < 18:
                temp_warning = (
                    "Low leaf temperature"
                )
                temp_status = (
                    "Cold Stress"
                )
                risk_level = "MEDIUM"
                action_priority = "MODERATE"
                temp_score = 10
                watering = (
                    "Reduce watering"
                )
                irrigation = (
                    "Avoid excess moisture"
                )
                thermal_action = (
                    "Protect plant from cold wind"
                )

            # Overwatering indication
            elif diff < -2:
                temp_warning = (
                    "Leaf cooler than ambient"
                )
                temp_status = (
                    "Possible overwatering"
                )
                risk_level = "LOW"
                action_priority = "LOW"
                temp_score = 5
                watering = (
                    "Reduce watering amount"
                )
                irrigation = (
                    "Check soil moisture"
                )
                thermal_action = (
                    "Allow soil drying"
                )

            # Disease onset
            elif diff > 5:
                temp_warning = (
                    "Leaf hotter than environment"
                )
                temp_status = (
                    "Possible disease onset"
                )
                risk_level = "MEDIUM"
                action_priority = "MODERATE"
                temp_score = 12
                watering = (
                    "Monitor moisture carefully"
                )
                irrigation = (
                    "Avoid wet leaves"
                )
                thermal_action = (
                    "Inspect leaf disease spread"
                )

        # =========================
        # CNN + TEMP FUSION
        # =========================

        cnn_score=confidence*100

        combined_score = (
            cnn_score * 0.8
        ) + (
            temp_score * 0.2
        )

        stress_index = 0
        if (
            leaf_temp is not None
            and ambient_temp is not None
        ):
            stress_index = min(
                100,
                round(
                    temp_score +
                    abs(diff * 3)
                )
            )

        if combined_score>90:

            final_risk="HIGH"

        elif combined_score>70:

            final_risk="MEDIUM"

        else:

            final_risk=risk_level

        if disease == "Tomato___healthy" and temp_score > 15:
            temp_status = "Thermal stress detected"
            final_risk = "MEDIUM"

        plant_recommendation = (
            watering
            + " | "
            + irrigation
            + " | "
            + thermal_action
        )

        print(
            f"""
            Disease : {disease}
            CNN : {cnn_score:.1f}
            Leaf Temp : {leaf_temp}
            Ambient : {ambient_temp}
            Risk : {risk_level}
            """
        )

        return jsonify({

            "disease":disease,

            "confidence":round(
                confidence*100,
                2
            ),

            "advice":
            info.get(
                "advice",
                "-"
            ),

            "fertilizer":
            info.get(
                "fertilizer",
                "-"
            ),

            "pesticide":
            info.get(
                "pesticide",
                "-"
            ),

            "organic":
            info.get(
                "organic",
                "-"
            ),

            "product_link":
            info.get(
                "product_link",
                "#"
            ),

            "youtube":
            info.get(
                "youtube",
                "#"
            ),

            "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

            "leaf_temp":
            leaf_temp if leaf_temp is not None else "N/A",

            "ambient_temp":
            ambient_temp if ambient_temp is not None else "N/A",

            # NEW FIELDS

            "temp_warning":
            temp_warning,

            "temp_status":
            temp_status,

            "risk_level":
            final_risk,

            "combined_score":
            round(
                combined_score,
                1
            ),

            "stress_index":
            stress_index,

            "temp_difference":
            round(diff, 1)
            if (
                leaf_temp is not None
                and ambient_temp is not None
            )
            else "N/A",

            "watering":
            watering,

            "irrigation":
            irrigation,

            "thermal_action":
            thermal_action,

            "action_priority":
            action_priority,

            "plant_recommendation":
            plant_recommendation,

            # NEW: Include the stored image path
            "image_path": filepath,

        })

    except Exception as e:

        import traceback

        traceback.print_exc()

        return jsonify({

            "error":
            str(e)

        }),500

    finally:
        # Images are now permanently stored in the uploads folder
        # No deletion happens here - images persist for future reference
        print(f"[IMAGE] Image stored permanently at: {filepath}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model is not None,
            "class_count": len(class_names),
            "sensor_data": sensor_data,
        }
    )


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🍅 LeafSense - Smart Plant Health Monitor")
    print("=" * 80)
    print(f"Serial Port: {SERIAL_PORT}")
    print(f"Baud Rate: {SERIAL_BAUD}")
    print(f"Serial Enabled: {SERIAL_ENABLED}")
    print("=" * 80 + "\n")

    # Start serial reader thread
    serial_thread = threading.Thread(target=serial_reader_worker, daemon=True)
    serial_thread.start()

    # Run Flask app
    local_ip = get_local_ip()
    print("[FLASK] Starting web server on:")
    print(f"         http://0.0.0.0:5000  (all interfaces)")
    print(f"         http://{local_ip}:5000  (local network)")

    if USE_NGROK:
        ngrok_url = start_ngrok(NGROK_PORT)
        if ngrok_url:
            print(f"         {ngrok_url}  (public ngrok tunnel)")
            print()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)