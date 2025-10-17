import re
import json
from word2number import w2n
from deep_translator import GoogleTranslator

# --- ElevenLabs API Key (replace with yours) ---
from elevenlabs import ElevenLabs

# Read the ElevenLabs API key from file
with open("keys/elevenlabs.key", "r") as f:
    API_KEY = f.read().strip()

# Initialize ElevenLabs client
client = ElevenLabs(api_key=API_KEY)


# --- Normalize Devanagari (Hindi/Gujarati) digits ---
def normalize_indic_digits(text):
    trans_table = str.maketrans("०१२३४५६७८९", "0123456789")
    return text.translate(trans_table)

# --- Step 1: Transcribe audio ---
def transcribe_audio(file_path, language_code="guj"):
    print(f"\n🎙️ Transcribing with language: {language_code} ...")
    with open(file_path, "rb") as audio:
        transcription = client.speech_to_text.convert(
            file=audio,
            model_id="scribe_v1",
            language_code=language_code
        )
    text = getattr(transcription, "text", str(transcription)).strip()
    print("✅ Transcription done.")
    return text, language_code

# --- Step 2: Translate to English ---
def translate_to_english(text):
    translator = GoogleTranslator(source="auto", target="en")
    return translator.translate(text)

# --- Step 3: Normalize numbers robustly ---
def normalize_numbers(text):
    text = normalize_indic_digits(text)
    text = text.lower()
    text = re.sub(r'[,\-–—()]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Fix multi-word number phrases like "forty five", "twenty eight"
    def replace_multiword_numbers(t):
        try:
            num = w2n.word_to_num(t)
            return str(num)
        except Exception:
            return t

    # Handle multi-word numeric sequences
    pattern = r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|' \
              r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|' \
              r'eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|' \
              r'eighty|ninety|hundred|thousand|million|and)(?:\s+(?:zero|one|two|' \
              r'three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|' \
              r'fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|' \
              r'thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|and))*\b'

    text = re.sub(pattern, lambda m: replace_multiword_numbers(m.group()), text)

    # Convert single-digit words (four one one three → 4113)
    digit_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9"
    }

    tokens = text.split()
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] in digit_words:
            digits = []
            while i < len(tokens) and tokens[i] in digit_words:
                digits.append(digit_words[tokens[i]])
                i += 1
            out.append("".join(digits))
        else:
            out.append(tokens[i])
            i += 1

    normalized = " ".join(out)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return normalized

# --- Step 4: Extract multiple pcode (5-char) + qty ---
def extract_multiple_pairs(normalized_text):
    clean = re.sub(r'[^a-z0-9\s]', ' ', normalized_text.lower())
    tokens = [t for t in clean.split() if t]
    results = []

    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if re.match(r'^[a-z]', tok):  # start of a product code
            letter = tok[0].upper()
            digits = ''.join(re.findall(r'\d+', tok))
            j = i + 1
            while j < len(tokens) and len(digits) < 4:
                if re.match(r'^\d+$', tokens[j]):
                    digits += tokens[j]
                else:
                    break
                j += 1

            if len(digits) >= 4:
                pcode = (letter + digits[:4]).upper()

                # find next numeric token as quantity
                k = j
                qty = None
                while k < len(tokens):
                    if re.match(r'^\d+$', tokens[k]):
                        qty = tokens[k]
                        break
                    k += 1

                if qty:
                    results.append({"pcode": pcode, "qty": qty})
                    i = k + 1
                    continue
        i += 1

    return results

# --- Step 5: Main Pipeline ---
def main(audio_file, language_code="eng"):
    try:
        original, lang = transcribe_audio(audio_file, language_code)
        print("\n🗣️ Original Transcription:\n", original)

        translated = translate_to_english(original)
        print("\n🌍 Translated to English:\n", translated)

        normalized = normalize_numbers(translated)
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
    main("c4113.mp3", language_code="eng")
