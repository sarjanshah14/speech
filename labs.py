import re
import json
from deep_translator import GoogleTranslator
from elevenlabs import ElevenLabs


# Read the ElevenLabs API key from file
with open("keys/elevenlabs.key", "r") as f:
    API_KEY = f.read().strip()

# Initialize ElevenLabs client
client = ElevenLabs(api_key=API_KEY)


# --- Normalize Devanagari digits ---
def normalize_indic_digits(text):
    return text.translate(str.maketrans("०१२३४५६७८९", "0123456789"))

# --- Transcribe audio ---
def transcribe_audio(file_path, language_code="eng"):
    print(f"\n🎙️ Transcribing with language: {language_code} ...")
    with open(file_path, "rb") as audio:
        transcription = client.speech_to_text.convert(
            file=audio,
            model_id="scribe_v1",
            language_code=language_code
        )
    text = getattr(transcription, "text", str(transcription)).strip()
    print("✅ Transcription done.")
    return text

# --- Translate to English ---
def translate_to_english(text):
    return GoogleTranslator(source="auto", target="en").translate(text)

# --- Convert spoken numbers to digits ---
def spoken_to_digits(text):
    text = normalize_indic_digits(text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    num_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
        "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
        "eighty": "80", "ninety": "90"
    }

    tokens = text.split()
    result = []
    i = 0

    while i < len(tokens):
        t = tokens[i]

        # Handle "forty five" type phrases
        if t in ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]:
            if i + 1 < len(tokens) and tokens[i + 1] in num_map and int(num_map[tokens[i + 1]]) < 10:
                result.append(str(int(num_map[t]) + int(num_map[tokens[i + 1]])))
                i += 2
                continue
            else:
                result.append(num_map[t])
                i += 1
                continue

        if t in num_map:
            result.append(num_map[t])
        else:
            result.append(t)
        i += 1

    return " ".join(result)

# --- Extract all pcode + qty pairs ---
def extract_multiple_pairs(normalized):
    stop_words = {"nos", "nang", "pieces", "pcs", "books", "qty", "numbers"}
    tokens = normalized.split()
    results = []

    i = 0
    while i < len(tokens):
        # find any token starting with a letter
        if re.match(r'^[a-z]', tokens[i]):
            code = tokens[i]
            digits = []
            j = i + 1

            # collect numbers until we reach 5 total chars (letters + digits)
            while j < len(tokens) and len(code + ''.join(digits)) < 5:
                if re.match(r'^\d+$', tokens[j]):
                    digits.append(tokens[j])
                else:
                    break
                j += 1

            pcode = (code + ''.join(digits))[:5].upper()

            # collect quantity (next standalone number)
            qty = ""
            while j < len(tokens):
                if tokens[j] in stop_words or re.match(r'^[a-z]', tokens[j]):
                    break
                if re.match(r'^\d+$', tokens[j]):
                    qty = tokens[j]
                j += 1

            if len(pcode) == 5 and qty:
                results.append({"pcode": pcode, "qty": qty})

            i = j
        else:
            i += 1

    return results

# --- Main pipeline ---
def main(audio_file, language_code="eng"):
    try:
        original = transcribe_audio(audio_file, language_code)
        print("\n🗣️ Original Transcription:\n", original)

        translated = translate_to_english(original)
        print("\n🌍 Translated to English:\n", translated)

        normalized = spoken_to_digits(translated)
        print("\n🧩 Normalized Numbers & Codes:\n", normalized)

        structured = extract_multiple_pairs(normalized)
        print("\n📦 Structured JSON Output:\n", json.dumps(structured, indent=2, ensure_ascii=False))

        with open("translation_output.txt", "w", encoding="utf-8") as f:
            f.write(f"Original Transcription:\n{original}\n\n")
            f.write(f"Translated to English:\n{translated}\n\n")
            f.write(f"Normalized Numbers & Codes:\n{normalized}\n\n")
            f.write(f"Structured JSON Output:\n{json.dumps(structured, indent=2, ensure_ascii=False)}\n")

        print("\n✅ Done! Saved translation_output.txt")

    except Exception as e:
        print("\n❌ Error:", e)

# --- Run ---
if __name__ == "__main__":
    main("try.mp3", language_code="guj")
