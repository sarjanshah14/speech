import os
import sys
import time
import tempfile
import wave
import re
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd

# Reuse transcriber from deep.py
from deep import ProductCodeTranscriber


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_DURATION = 2.0
SILENCE_SECONDS = 1.5
SILENCE_AMPLITUDE = 300


class LiveTranscriber:
    def __init__(self, api_key: str, excel_file: str = "productlist.xlsx") -> None:
        self.api_key = api_key
        self.should_stop = False
        
        # Load parser from deep.py
        excel_path = os.path.join(os.path.dirname(__file__), excel_file)
        self.parser = ProductCodeTranscriber(api_key, excel_file=excel_path)
        
        # Audio buffer
        self.audio_buffer = []
        self.last_audio_ts: float = time.time()
        self.is_recording = False
        
        # Comprehensive similar sounding character mappings for confusion handling
        self.similar_chars = {
            'a': ['e', 'o'],
            'b': ['d', 'p', 'v'],
            'c': ['s', 'k', 'g'],
            'd': ['b', 't'],
            'e': ['a', 'i'],
            'f': ['v', 'p', 's'],
            'g': ['j', 'k', 'c'],
            'h': ['n'],  # h can be silent or confused with n
            'i': ['e', 'y'],
            'j': ['g'],
            'k': ['c', 'g', 'q'],
            'l': ['r'],
            'm': ['n'],
            'n': ['m'],
            'o': ['a', 'u'],
            'p': ['b', 'f'],
            'q': ['k'],
            'r': ['l'],
            's': ['f', 'c', 'z'],
            't': ['d'],
            'u': ['o'],
            'v': ['f', 'w'],
            'w': ['v'],
            'x': ['z'],
            'y': ['i'],
            'z': ['s', 'x'],
        }
    
    def _silence_detector(self, audio_data: np.ndarray) -> bool:
        """Detect if audio is silence"""
        if audio_data.size == 0:
            return True
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        return rms < SILENCE_AMPLITUDE
    
    def _audio_callback(self, indata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            pass  # Suppress overflow messages
        
        now = time.time()
        self.audio_buffer.extend(indata.flatten())
        
        if self._silence_detector(indata):
            if self.is_recording and (now - self.last_audio_ts) >= SILENCE_SECONDS:
                self._process_audio_chunk()
                self.audio_buffer = []
                self.is_recording = False
        else:
            self.is_recording = True
            self.last_audio_ts = now
    
    def normalize_dialect_variants(self, text: str) -> str:
        """
        Handle different dialect pronunciations.
        Examples:
            "six thousand eighty" -> "sixty eighty"
            "p c" -> "pieces"
        """
        text = text.lower().strip()
        
        # Replace common transcription errors
        text = re.sub(r'\bp\s+c\b', 'pieces', text)
        text = re.sub(r'\bpc\b', 'pieces', text)
        
        # "six thousand eighty" pattern -> "sixty eighty"
        # Pattern: X thousand Y -> X*10 Y (where X is single digit)
        pattern = r'\b(one|two|three|four|five|six|seven|eight|nine)\s+thousand\s+(ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)'
        
        def replace_thousand(match):
            first = match.group(1)
            second = match.group(2)
            
            # "six thousand eighty" -> "sixty eighty"
            tens_map = {
                'one': 'ten', 'two': 'twenty', 'three': 'thirty', 'four': 'forty',
                'five': 'fifty', 'six': 'sixty', 'seven': 'seventy', 'eight': 'eighty', 'nine': 'ninety'
            }
            return f"{tens_map[first]} {second}"
        
        text = re.sub(pattern, replace_thousand, text)
        return text
    
    def find_similar_pcodes(self, pcode: str) -> List[str]:
        """
        Find similar product codes by replacing confused characters.
        E.g., if heard "b0058" but list has "d0058", find it.
        Also checks if transcribed "s15801" but should be "f15801"
        """
        if not pcode:
            return []
        
        candidates = [pcode.lower()]
        
        # Generate variants by replacing each character with similar ones
        for i, char in enumerate(pcode.lower()):
            if char in self.similar_chars:
                for similar in self.similar_chars[char]:
                    variant = pcode[:i].lower() + similar + pcode[i+1:].lower()
                    candidates.append(variant)
        
        # Check which candidates exist in product list
        found = []
        for candidate in candidates:
            for pcode_in_list in self.parser.pcode_list:
                if str(pcode_in_list).lower() == candidate:
                    found.append(str(pcode_in_list).lower())
                    if candidate != pcode.lower():
                        print(f"  ⚠️  Character confusion corrected: '{pcode}' → '{candidate}'")
                    break
        
        return found if found else [pcode.lower()]
    
    def extract_pcode_with_variants(self, transcript: str) -> Tuple[str, str]:
        """
        Extract pcode handling dialect variants and similar-sounding characters.
        Handles "dot" as part of product codes with dots.
        Returns (pcode, quantity)
        """
        # First normalize dialect variants
        normalized_transcript = self.normalize_dialect_variants(transcript)

        # Handle "dot" as separator for codes with dots
        if " dot " in normalized_transcript.lower():
            normalized_transcript = normalized_transcript.lower().replace(" dot ", ".")

        # Try standard extraction from deep.py
        pcode, qty, _ = self.parser.extract_pcode_and_qty(normalized_transcript)
        
        if pcode:
            # Check if exact match exists
            found_exact = False
            for p in self.parser.pcode_list:
                if str(p).lower() == pcode.lower():
                    found_exact = True
                    break
            
            if not found_exact:
                # Try similar sounding variants
                similar = self.find_similar_pcodes(pcode)
                if similar and similar[0] != pcode.lower():
                    # Found a better match
                    old_pcode = pcode
                    pcode = similar[0]
                    
                    # Recalculate quantity after correcting pcode
                    # Re-extract with corrected pcode
                    normalized = self.parser.normalize_pcode_portion(normalized_transcript)
                    matches = re.findall(r'[a-z]+\d+', normalized.replace(' ', ''))
                    if matches:
                        extracted_sequence = matches[0]
                        if extracted_sequence.startswith(pcode):
                            qty = extracted_sequence[len(pcode):]
        
        return pcode, qty
    
    def _process_audio_chunk(self) -> None:
        """Process a chunk of audio by transcribing it"""
        if not self.audio_buffer:
            return
        
        audio_data = np.array(self.audio_buffer, dtype=np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Write WAV file
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_data.tobytes())
            
            # Transcribe
            transcript_result = self.parser.transcribe_audio(temp_filename)
            
            # Extract transcript text
            transcript_text = ""
            if 'results' in transcript_result and 'channels' in transcript_result['results']:
                channel = transcript_result['results']['channels'][0]
                if 'alternatives' in channel:
                    transcript_text = channel['alternatives'][0].get('transcript', '')
            
            if transcript_text.strip():
                print(f"🎤 Live: {transcript_text}")
                
                try:
                    # Extract with dialect and confusion handling
                    pcode, qty = self.extract_pcode_with_variants(transcript_text)
                    
                    if pcode:
                        # Verify pcode exists in list
                        pcode_exists = False
                        for p in self.parser.pcode_list:
                            if str(p).lower() == pcode.lower():
                                pcode_exists = True
                                pcode = str(p)  # Use exact case from list
                                break
                        
                        if pcode_exists:
                            print(f"✅ Product: {pcode.upper()} | Qty: {qty or '-'}")
                        else:
                            print(f"❌ Product code '{pcode}' not found in list")
                    else:
                        print("❌ Can't recognize product code")
                        
                except Exception as e:
                    print(f"❌ Parse error: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("🎤 Live: [silence or no speech detected]")
        
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                os.unlink(temp_filename)
            except:
                pass
    
    def run(self) -> None:
        print("🎤 Listening... Press CTRL+C to stop.")
        
        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=self._audio_callback,
                blocksize=int(SAMPLE_RATE * 0.1)
            ):
                while not self.should_stop:
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n🛑 Stopped listening.")
        except Exception as e:
            print(f"❌ Audio error: {e}")


def main() -> None:
    api_key_path_default = os.path.join(os.path.dirname(__file__), "keys", "deepgram.key")
    api_key = os.environ.get("DEEPGRAM_API_KEY")
    
    if not api_key:
        key_path = os.environ.get("DEEPGRAM_KEY_PATH", api_key_path_default)
        try:
            with open(key_path, "r") as f:
                api_key = f.read().strip()
        except Exception:
            # Fallback to hardcoded key
            api_key = "c3ddb4025050ecf0030421c7a8e2d8d04656ad7f"
    
    transcriber = LiveTranscriber(api_key)
    transcriber.run()


if __name__ == "__main__":
    main()