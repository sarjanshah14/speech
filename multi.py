# 312519014593
import os
import re
from difflib import get_close_matches
from pydub import AudioSegment
from google.cloud import speech_v2
from google.cloud import translate_v2 as translate
import pandas as pd
from word2number import w2n

# ===========================
# CONFIG
# ===========================
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keys/vigilant-art-475111-c1-cd0522c17ae2.json"
PRODUCT_XLSX = "productlist.xlsx"

# ===========================
# LOAD PRODUCT CODES
# ===========================
products_df = pd.read_excel(PRODUCT_XLSX)
pcodes = products_df.iloc[:, 0].astype(str).tolist()

# ===========================
# HELPERS
# ===========================
def convert_to_wav(input_file):
    """Ensure audio is in WAV format."""
    if not input_file.lower().endswith(".wav"):
        print(f"🎵 Converting {input_file} → WAV ...")
        sound = AudioSegment.from_file(input_file)
        wav_path = input_file.rsplit(".", 1)[0] + ".wav"
        sound.export(wav_path, format="wav")
        return wav_path
    return input_file


def normalize_text(txt):
    """Clean and normalize text."""
    txt = txt.lower()
    replacements = {
        "see": "c", "sea": "c", "si": "c", "she": "c", "cee": "c",
        "are": "r", "ar": "r", "our": "r",
        "tea": "t", "ti": "t", "tee": "t", "dee": "d",
        "b": "b", "p": "p"
    }
    for k, v in replacements.items():
        txt = re.sub(rf"\b{k}\b", v, txt)
    txt = re.sub(r"[^a-z0-9 ]+", " ", txt)
    return txt.strip()


def repair_codes(text):
    """Fix misheard code patterns."""
    # handle Gujarati/Hindi 'સી' or 'श्री' → 'C'
    text = re.sub(r"(સી|श्री)", "c", text, flags=re.IGNORECASE)
    # join separated 'c 4042' → 'c4042'
    text = re.sub(r"\bc\s*([0-9]{3,5})", r"c\1", text)
    text = re.sub(r"\br\s*([0-9]{3,5})", r"r\1", text)
    text = re.sub(r"\bt\s*([0-9]{3,5})", r"t\1", text)
    return text


def words_to_numbers(text):
    """Convert number words to digits if possible."""
    words = text.split()
    converted = []
    for w in words:
        try:
            converted.append(str(w2n.word_to_num(w)))
        except:
            converted.append(w)
    return " ".join(converted)


def find_pcodes_and_qty(text):
    """Extract product codes and quantities."""
    pattern = r"([a-z]{1,3}\d{2,5})\s*(\d+)"
    return re.findall(pattern, text, re.IGNORECASE)


def fuzzy_match(pcode):
    match = get_close_matches(pcode, pcodes, n=1, cutoff=0.6)
    return match[0] if match else pcode


# ===========================
# SPEECH-TO-TEXT v2
# ===========================
def transcribe_v2(audio_path, lang):
    client = speech_v2.SpeechClient()
    audio_path = convert_to_wav(audio_path)

    with open(audio_path, "rb") as f:
        audio_content = f.read()

    config = speech_v2.RecognitionConfig(
        auto_decoding_config=speech_v2.AutoDetectDecodingConfig(),
        language_codes=[lang],
        model="latest_long",
        features=speech_v2.RecognitionFeatures(enable_word_time_offsets=True)
    )

    # Use default recognizer (auto-created)
    recognizer = "projects/312519014593/locations/global/recognizers/_"

    request = speech_v2.RecognizeRequest(
        recognizer=recognizer,
        config=config,
        content=audio_content
    )

    response = client.recognize(request=request)
    transcript = " ".join([r.alternatives[0].transcript for r in response.results])
    return transcript


# ===========================
# MAIN PROCESS
# ===========================
def process_audio(file_path):
    lang = input("🎙️ Which language are you speaking (e.g., en-IN, hi-IN, gu-IN)? ").strip() or "en-IN"
    print(f"🧠 Using language: {lang}")

    print("🔊 Transcribing audio ...")
    raw_text = transcribe_v2(file_path, lang)
    print("🗣️ Raw Transcript:", raw_text)

    # Try translation only if not English
    if not lang.startswith("en"):
        translate_client = translate.Client()
        translated = translate_client.translate(raw_text, target_language="en")["translatedText"]
        print("🌐 Translated to English:", translated)
    else:
        translated = raw_text

    # Normalization pipeline
    text = words_to_numbers(translated)
    text = repair_codes(text)
    text = normalize_text(text)
    print("🧾 Normalized:", text)

    pairs = find_pcodes_and_qty(text)
    if not pairs:
        print("❌ No product code patterns found.")
        return

    print("\n✅ Final Results:")
    for i, (pcode, qty) in enumerate(pairs, 1):
        matched = fuzzy_match(pcode)
        print(f"{i}. Code → {matched} | Qty → {qty}")


# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    path = input("🎤 Enter audio path (e.g., audio/try.mp3): ").strip()
    if not os.path.exists(path):
        print("❌ File not found.")
    else:
        process_audio(path)
