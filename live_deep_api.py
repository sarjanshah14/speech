import os
import io
import sounddevice as sd
import numpy as np
import wave
import requests
from flask import Flask, jsonify
import openpyxl
import re
from difflib import get_close_matches

# =====================================================
# === CONFIGURATION ===
# =====================================================

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
if not DEEPGRAM_API_KEY and os.path.exists("keys/deepgram.key"):
    with open("keys/deepgram.key") as f:
        DEEPGRAM_API_KEY = f.read().strip()

if not DEEPGRAM_API_KEY:
    raise Exception("❌ Deepgram API key not found. Set env var or keys/deepgram.key")

DEEPGRAM_URL = "https://api.deepgram.com/v1/listen?model=nova-2&smart_format=true"

# Load Product List
PRODUCT_FILE = "productlist.xlsx"
if not os.path.exists(PRODUCT_FILE):
    raise FileNotFoundError(f"{PRODUCT_FILE} not found!")

wb = openpyxl.load_workbook(PRODUCT_FILE)
sheet = wb.active
PRODUCTS = [str(cell.value).strip() for cell in sheet['A'] if cell.value]

# Flask App
app = Flask(__name__)

# =====================================================
# === FUNCTION: Record Audio and Transcribe ===
# =====================================================

def record_and_transcribe_once(duration=5):
    """Records mic input, sends to Deepgram REST API"""
    print("🎙️ Listening for speech...")
    fs = 16000
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("✅ Recording complete, sending to Deepgram...")

    # Convert to WAV bytes
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio.tobytes())
    buf.seek(0)

    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}",
        "Content-Type": "audio/wav"
    }

    try:
        response = requests.post(DEEPGRAM_URL, headers=headers, data=buf.read())
        response.raise_for_status()
        data = response.json()
        transcript = data['results']['channels'][0]['alternatives'][0]['transcript']
        print(f"🧾 Transcript: {transcript}")
        return transcript

    except Exception as e:
        print(f"❌ Deepgram transcription error: {e}")
        return None

# =====================================================
# === FUNCTION: Parse Product + Quantity ===
# =====================================================

def parse_product_and_quantity(transcript):
    """Extracts product code and quantity from transcript using fuzzy match"""
    if not transcript:
        return None, None

    numbers = re.findall(r'\d+', transcript)
    quantity = numbers[0] if numbers else None

    matches = get_close_matches(transcript.upper(), PRODUCTS, n=1, cutoff=0.6)
    product = matches[0] if matches else None

    return product, quantity

# =====================================================
# === ROUTE: /transcribe ===
# =====================================================

@app.route('/transcribe', methods=['GET'])
def transcribe_once():
    transcript = record_and_transcribe_once()
    if not transcript:
        return jsonify({"error": "Transcription failed"}), 500

    product, quantity = parse_product_and_quantity(transcript)
    result = {
        "transcript": transcript,
        "product": product or "Unknown",
        "quantity": quantity or "Unknown"
    }
    print("🧩 Parsed:", result)
    return jsonify(result)

# =====================================================
# === MAIN ENTRY POINT ===
# =====================================================

if __name__ == '__main__':
    print("🚀 live_deep_api running on http://localhost:5000/transcribe")
    app.run(host="0.0.0.0", port=5000)
