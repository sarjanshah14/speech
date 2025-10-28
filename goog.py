#!/usr/bin/env python3
"""
Enhanced Live Google Speech-to-Text Transcriber
Fixed pattern recognition for product codes with proper quantity extraction
"""

import os
import sys
import time
import tempfile
import wave
import re
from typing import Tuple, List
from difflib import SequenceMatcher

import numpy as np
import sounddevice as sd
import openpyxl
from google.cloud import speech

# Set Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"C:\Users\itintern2\Desktop\project1\keys\vigilant-art-475111-c1-cd0522c17ae2.json"

# AUDIO PARAMETERS
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
SILENCE_SECONDS = 1.5
SILENCE_AMPLITUDE = 300


class LiveTranscriber:
    def __init__(self, excel_file: str = "productlist.xlsx", language_code: str = "en-US"):
        self.should_stop = False
        self.language_code = language_code
        
        # Load product codes from Excel
        excel_path = os.path.join(os.path.dirname(__file__), excel_file)
        self.pcode_list = self.load_pcodes_from_excel(excel_path)
        self.pcode_list_no_dots = [str(p).replace('.', '').lower() for p in self.pcode_list]
        print(f"✅ Loaded {len(self.pcode_list)} product codes")
        
        # Audio buffer
        self.audio_buffer = []
        self.last_audio_ts: float = time.time()
        self.is_recording = False
        
        # Google Cloud Speech client
        self.client = speech.SpeechClient()
        
        # Similar sounding character mappings
        self.similar_chars = {
            'a': ['e', 'o'],
            'b': ['d', 'p', 'v'],
            'c': ['s', 'k', 'g'],
            'd': ['b', 't'],
            'e': ['a', 'i'],
            'f': ['v', 'p', 's'],
            'g': ['j', 'k', 'c'],
            'h': ['n', '8'],
            'i': ['e', 'y', '1'],
            'j': ['g'],
            'k': ['c', 'g', 'q'],
            'l': ['r', '1'],
            'm': ['n'],
            'n': ['m', 'h'],
            'o': ['a', 'u', '0'],
            'p': ['b', 'f'],
            'q': ['k'],
            'r': ['l'],
            's': ['f', 'c', 'z', '5'],
            't': ['d'],
            'u': ['o'],
            'v': ['f', 'w', 'b'],
            'w': ['v', 'u'],
            'x': ['z'],
            'y': ['i'],
            'z': ['s', 'x'],
            '0': ['o'],
            '1': ['i', 'l'],
            '5': ['s'],
        }
    
    def load_pcodes_from_excel(self, excel_file: str) -> List[str]:
        """Load product codes from Excel file first column."""
        try:
            wb = openpyxl.load_workbook(excel_file)
            ws = wb.active
            
            pcodes = []
            for row in ws.iter_rows(min_row=1, max_col=1, values_only=True):
                if row[0]:
                    code = str(row[0]).strip()
                    pcodes.append(code)
            
            wb.close()
            return pcodes
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            return []
    
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
        self.audio_buffer.extend(indata.flatten().tolist())
        
        if self._silence_detector(indata):
            if self.is_recording and (now - self.last_audio_ts) >= SILENCE_SECONDS:
                self._process_audio_chunk()
                self.audio_buffer = []
                self.is_recording = False
        else:
            self.is_recording = True
            self.last_audio_ts = now
    
    def normalize_number_text(self, text: str) -> str:
        """Convert spelled-out numbers to digits with proper double/triple handling."""
        text = text.lower().strip()
        
        print(f"  [DEBUG] Original: '{text}'")
        
        # Remove punctuation except dots and spaces
        text = re.sub(r'[,!?;:\']', '', text)
        
        # Common transcription fixes
        text = re.sub(r'\bp\s+c\b', 'pieces', text)
        text = re.sub(r'\bpc\b', 'pieces', text)
        text = re.sub(r'\bzed\b', 'z', text)
        text = re.sub(r'\band\b', ' ', text)
        
        # Fix common misheard words
        text = re.sub(r'\bcome\b', 'cum', text)
        text = re.sub(r'\bpace\b', 'piece', text)
        text = re.sub(r'\bpage\b', 'piece', text)
        
        # CRITICAL FIX: Only convert "in" to "n" in very specific cases
        # Check if it's "in" followed by known product code prefixes
        words = text.split()
        if len(words) >= 2 and words[0] == 'in':
            # Only convert if followed by specific patterns that suggest it's a product code
            next_part = words[1]
            # Check if it starts with common IN-prefix patterns: c, b, t, i (for INC, INB, INT, INI, etc.)
            if next_part and len(next_part) >= 2 and next_part[0] in ['c', 'b', 't', 'i', 'u']:
                # Additional check: the second word should be mostly letters
                if re.match(r'^[a-z]{2,}', next_part):
                    words[0] = 'n'
                    text = ' '.join(words)
                    print(f"  [DEBUG] Converted 'in' → 'n' (detected product code pattern)")
        
        print(f"  [DEBUG] After fixes: '{text}'")
        
        # Handle "dot" patterns BEFORE number conversions
        text = re.sub(r'\bdots?\s+dots?\b', '..', text)
        text = re.sub(r'\bdots?\b', '.', text)
        
        # CRITICAL FIX: Only process double/triple when explicitly mentioned
        number_words = {
            'zero': '0', 'oh': '0',
            'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
        }
        
        # Only if "double" or "triple" exists in text
        if 'double' in text or 'triple' in text:
            print(f"  [DEBUG] Found double/triple keyword")
            
            # Handle "X double Y" -> "XYY"
            for num1_word, num1_digit in number_words.items():
                for num2_word, num2_digit in number_words.items():
                    pattern = rf'\b{num1_word}\s+double\s+{num2_word}\b'
                    replacement = num1_digit + num2_digit + num2_digit
                    text = re.sub(pattern, replacement, text)
            
            # Handle "X triple Y" -> "XYYY"
            for num1_word, num1_digit in number_words.items():
                for num2_word, num2_digit in number_words.items():
                    pattern = rf'\b{num1_word}\s+triple\s+{num2_word}\b'
                    replacement = num1_digit + num2_digit + num2_digit + num2_digit
                    text = re.sub(pattern, replacement, text)
            
            # Handle standalone "double X" -> "XX"
            for word, digit in number_words.items():
                pattern = rf'\bdouble\s+{word}\b'
                text = re.sub(pattern, digit + digit, text)
            
            # Handle standalone "triple X" -> "XXX"
            for word, digit in number_words.items():
                pattern = rf'\btriple\s+{word}\b'
                text = re.sub(pattern, digit + digit + digit, text)
        
        print(f"  [DEBUG] After double/triple: '{text}'")
        
        # Compound numbers
        compound_patterns = [
            (r'\btwenty\s+one\b', '21'), (r'\btwenty\s+two\b', '22'), (r'\btwenty\s+three\b', '23'),
            (r'\btwenty\s+four\b', '24'), (r'\btwenty\s+five\b', '25'), (r'\btwenty\s+six\b', '26'),
            (r'\btwenty\s+seven\b', '27'), (r'\btwenty\s+eight\b', '28'), (r'\btwenty\s+nine\b', '29'),
            (r'\bthirty\s+one\b', '31'), (r'\bthirty\s+two\b', '32'), (r'\bthirty\s+three\b', '33'),
            (r'\bthirty\s+four\b', '34'), (r'\bthirty\s+five\b', '35'), (r'\bthirty\s+six\b', '36'),
            (r'\bthirty\s+seven\b', '37'), (r'\bthirty\s+eight\b', '38'), (r'\bthirty\s+nine\b', '39'),
            (r'\bforty\s+one\b', '41'), (r'\bforty\s+two\b', '42'), (r'\bforty\s+three\b', '43'),
            (r'\bforty\s+four\b', '44'), (r'\bforty\s+five\b', '45'), (r'\bforty\s+six\b', '46'),
            (r'\bforty\s+seven\b', '47'), (r'\bforty\s+eight\b', '48'), (r'\bforty\s+nine\b', '49'),
            (r'\bfifty\s+one\b', '51'), (r'\bfifty\s+two\b', '52'), (r'\bfifty\s+three\b', '53'),
            (r'\bfifty\s+four\b', '54'), (r'\bfifty\s+five\b', '55'), (r'\bfifty\s+six\b', '56'),
            (r'\bfifty\s+seven\b', '57'), (r'\bfifty\s+eight\b', '58'), (r'\bfifty\s+nine\b', '59'),
            (r'\bsixty\s+one\b', '61'), (r'\bsixty\s+two\b', '62'), (r'\bsixty\s+three\b', '63'),
            (r'\bsixty\s+four\b', '64'), (r'\bsixty\s+five\b', '65'), (r'\bsixty\s+six\b', '66'),
            (r'\bsixty\s+seven\b', '67'), (r'\bsixty\s+eight\b', '68'), (r'\bsixty\s+nine\b', '69'),
            (r'\bseventy\s+one\b', '71'), (r'\bseventy\s+two\b', '72'), (r'\bseventy\s+three\b', '73'),
            (r'\bseventy\s+four\b', '74'), (r'\bseventy\s+five\b', '75'), (r'\bseventy\s+six\b', '76'),
            (r'\bseventy\s+seven\b', '77'), (r'\bseventy\s+eight\b', '78'), (r'\bseventy\s+nine\b', '79'),
            (r'\beighty\s+one\b', '81'), (r'\beighty\s+two\b', '82'), (r'\beighty\s+three\b', '83'),
            (r'\beighty\s+four\b', '84'), (r'\beighty\s+five\b', '85'), (r'\beighty\s+six\b', '86'),
            (r'\beighty\s+seven\b', '87'), (r'\beighty\s+eight\b', '88'), (r'\beighty\s+nine\b', '89'),
            (r'\bninety\s+one\b', '91'), (r'\bninety\s+two\b', '92'), (r'\bninety\s+three\b', '93'),
            (r'\bninety\s+four\b', '94'), (r'\bninety\s+five\b', '95'), (r'\bninety\s+six\b', '96'),
            (r'\bninety\s+seven\b', '97'), (r'\bninety\s+eight\b', '98'), (r'\bninety\s+nine\b', '99'),
        ]
        
        for pattern, replacement in compound_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Simple digit mapping
        word_to_digit = {
            'zero': '0', 'oh': '0',
            'one': '1',
            'two': '2', 'to': '2', 'too': '2',
            'three': '3',
            'four': '4', 'for': '4',
            'five': '5',
            'six': '6',
            'seven': '7',
            'eight': '8', 'ate': '8',
            'nine': '9',
            'ten': '10',
            'eleven': '11',
            'twelve': '12',
            'thirteen': '13',
            'fourteen': '14',
            'fifteen': '15',
            'sixteen': '16',
            'seventeen': '17',
            'eighteen': '18',
            'nineteen': '19',
            'twenty': '20',
            'thirty': '30',
            'forty': '40',
            'fifty': '50',
            'sixty': '60',
            'seventy': '70',
            'eighty': '80',
            'ninety': '90',
        }
        
        for word, digit in word_to_digit.items():
            text = re.sub(r'\b' + word + r'\b', digit, text)
        
        print(f"  [DEBUG] After digit mapping: '{text}'")
        
        # Keep letters, digits, dots, and spaces
        text = re.sub(r'[^\w\s.]', '', text)
        
        return text
    
    def extract_candidate_sequences(self, normalized: str, original: str) -> List[str]:
        """Extract all possible product code sequences from normalized text."""
        candidates = []
        
        # Remove extra spaces
        clean_text = ' '.join(normalized.split())
        no_space = clean_text.replace(' ', '')
        
        # CRITICAL FIX: Extract numbers with spaces as potential codes
        # Example: "35 335" should become "35335"
        # Look for patterns like: digit+ space digit+
        spaced_numbers = re.findall(r'\d+(?:\s+\d+)+', clean_text)
        for match in spaced_numbers:
            compact = match.replace(' ', '')
            if len(compact) >= 4:  # At least 4 digits
                candidates.append(compact)
                print(f"  [DEBUG] Spaced number candidate: '{match}' → '{compact}'")
        
        # Pattern 1: Codes with dots (S1R.SS.E, JC.TM.NPP, etc.)
        dot_patterns = re.findall(r'[a-z]+\d*[a-z]*\.[a-z0-9.&]+', no_space)
        candidates.extend(dot_patterns)
        
        # Pattern 2: Letter(s) + digit(s)
        matches = re.findall(r'[a-z]{1,15}\d+', no_space)
        candidates.extend(matches)
        
        # Pattern 3: Also check with single space preserved
        spaced_matches = re.findall(r'[a-z]\s+[a-z]\d+', clean_text)
        candidates.extend([m.replace(' ', '') for m in spaced_matches])
        
        # Pattern 4: Pure numeric codes (4+ digits)
        pure_numeric = re.findall(r'\d{4,}', no_space)
        candidates.extend(pure_numeric)
        
        print(f"  [DEBUG] Candidates: {candidates}")
        return candidates
    
    def fuzzy_match_with_dots(self, candidate: str) -> Tuple[str, str]:
        """Try to match candidate against pcodes with dots removed. Returns (matched_pcode, remainder)"""
        candidate_clean = candidate.replace('.', '').replace(' ', '').lower()
        
        for i, pcode_no_dot in enumerate(self.pcode_list_no_dots):
            # Exact match
            if candidate_clean == pcode_no_dot:
                print(f"  [DEBUG] Fuzzy dot match: '{candidate}' → '{self.pcode_list[i]}'")
                return str(self.pcode_list[i]), ""
            
            # Prefix match
            if candidate_clean.startswith(pcode_no_dot) and len(pcode_no_dot) >= 3:
                remainder = candidate_clean[len(pcode_no_dot):]
                print(f"  [DEBUG] Fuzzy dot prefix: '{candidate}' → '{self.pcode_list[i]}' (remainder: {remainder})")
                return str(self.pcode_list[i]), remainder
        
        return "", ""
    
    def find_best_pcode_match(self, candidates: List[str]) -> Tuple[str, str]:
        """Find the best matching product code from candidates."""
        best_match = ""
        best_length = 0
        remainder = ""
        
        for candidate in candidates:
            candidate_lower = candidate.lower()
            
            # Try exact match first
            for pcode in self.pcode_list:
                pcode_clean = str(pcode).lower()
                
                # Exact match
                if candidate_lower == pcode_clean:
                    if len(pcode_clean) >= best_length:
                        best_length = len(pcode_clean)
                        best_match = str(pcode)
                        remainder = ""
                        print(f"  [DEBUG] Exact match: {best_match}")
                
                # Candidate starts with pcode
                elif candidate_lower.startswith(pcode_clean):
                    if len(pcode_clean) > best_length:
                        best_length = len(pcode_clean)
                        best_match = str(pcode)
                        remainder = candidate_lower[len(pcode_clean):]
                        print(f"  [DEBUG] Prefix match: {best_match} (remainder: {remainder})")
        
        # If no match, try fuzzy matching without dots
        if not best_match:
            for candidate in candidates:
                fuzzy_pcode, fuzzy_remainder = self.fuzzy_match_with_dots(candidate)
                if fuzzy_pcode:
                    if len(fuzzy_pcode.replace('.', '')) > best_length:
                        best_match = fuzzy_pcode
                        remainder = fuzzy_remainder
                        best_length = len(fuzzy_pcode.replace('.', ''))
        
        # Try character confusion correction
        if not best_match:
            for candidate in candidates:
                similar_candidates = self.find_similar_pcodes(candidate)
                if similar_candidates:
                    best_match = similar_candidates[0]
                    remainder = ""
                    break
        
        return best_match, remainder
    
    def find_similar_pcodes(self, pcode: str) -> List[str]:
        """Find similar product codes by replacing confused characters."""
        if not pcode:
            return []
        
        candidates = [pcode.lower()]
        
        # Generate variants by replacing similar characters
        for i, char in enumerate(pcode.lower()):
            if char in self.similar_chars:
                for similar in self.similar_chars[char]:
                    variant = pcode[:i].lower() + similar + pcode[i+1:].lower()
                    candidates.append(variant)
        
        found = []
        for candidate in candidates:
            # Check both with and without dots
            candidate_no_dot = candidate.replace('.', '')
            
            for i, pcode_in_list in enumerate(self.pcode_list):
                pcode_lower = str(pcode_in_list).lower()
                pcode_no_dot = self.pcode_list_no_dots[i]
                
                if pcode_lower == candidate or pcode_no_dot == candidate_no_dot:
                    found.append(str(pcode_in_list))
                    if candidate != pcode.lower():
                        print(f"  [DEBUG] ⚠️  Confusion fix: '{pcode}' → '{candidate}' = {pcode_in_list}")
                    break
        
        return found
    
    def extract_quantity_from_remainder(self, normalized: str, pcode_match: str, remainder: str) -> str:
        """Extract quantity from remainder or from the normalized text after the product code."""
        qty = ""
        
        # If we have a remainder, use it as quantity
        if remainder and remainder.isdigit():
            qty = remainder
            print(f"  [DEBUG] Quantity from remainder: {qty}")
            return qty
        
        # Otherwise, look for numbers after the product code in the normalized text
        pcode_clean = pcode_match.replace('.', '').lower()
        normalized_clean = normalized.replace(' ', '').lower()
        
        # Find where the product code ends in the normalized text
        idx = normalized_clean.find(pcode_clean)
        if idx != -1:
            after_pcode = normalized_clean[idx + len(pcode_clean):]
            # Extract first number sequence after the product code
            qty_match = re.search(r'(\d+)', after_pcode)
            if qty_match:
                qty = qty_match.group(1)
                print(f"  [DEBUG] Quantity found after pcode: {qty}")
        
        return qty
    
    def extract_pcode_and_qty(self, transcript: str) -> Tuple[str, str]:
        """Extract pcode and quantity with enhanced pattern recognition."""
        text_lower = transcript.lower().strip()
        
        # Special case: "eight" at start might be "H"
        words = text_lower.split()
        if words and words[0] == 'eight' and len(words) > 1:
            digit_words = {'zero', 'oh', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'}
            if words[1] in digit_words:
                text_lower = 'h ' + ' '.join(words[1:])
                print(f"  [DEBUG] Corrected 'eight' → 'h': '{text_lower}'")
        
        # Normalize text
        normalized = self.normalize_number_text(text_lower)
        print(f"  [DEBUG] Normalized: '{normalized}'")
        
        # Extract all candidate sequences
        candidates = self.extract_candidate_sequences(normalized, transcript)
        
        if not candidates:
            print(f"  [DEBUG] ❌ No candidates found")
            return "", ""
        
        # Find best match
        best_match, remainder = self.find_best_pcode_match(candidates)
        
        # Extract quantity
        qty = self.extract_quantity_from_remainder(normalized, best_match, remainder)
        
        print(f"  [DEBUG] ✓ Result: '{best_match}' | Qty: '{qty}'")
        return best_match, qty
    
    def _process_audio_chunk(self) -> None:
        """Process audio chunk using Google Speech-to-Text"""
        if not self.audio_buffer:
            return
        
        audio_data = np.array(self.audio_buffer, dtype=np.int16)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name
        
        try:
            # Write WAV
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_data.tobytes())
            
            # Send to Google
            with open(temp_filename, 'rb') as f:
                content = f.read()
            
            audio = speech.RecognitionAudio(content=content)
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=SAMPLE_RATE,
                language_code=self.language_code,
                enable_automatic_punctuation=False,
                use_enhanced=True,
                model='default',
            )
            
            response = self.client.recognize(config=config, audio=audio)
            
            transcript_text = ""
            if response.results:
                transcript_text = " ".join(r.alternatives[0].transcript for r in response.results).strip()
            
            if transcript_text.strip():
                print(f"\n{'='*70}")
                print(f"🎤 LIVE: {transcript_text}")
                print(f"{'='*70}")
                
                try:
                    pcode, qty = self.extract_pcode_and_qty(transcript_text)
                    
                    if pcode:
                        print(f"\n{'─'*70}")
                        print(f"✅ PRODUCT: {pcode.upper()}  |  QTY: {qty or '-'}")
                        print(f"{'─'*70}\n")
                    else:
                        print(f"\n{'─'*70}")
                        print(f"❌ PRODUCT: Not recognized")
                        print(f"{'─'*70}\n")
                
                except Exception as e:
                    print(f"\n❌ Parse error: {e}\n")
            else:
                print(f"\n🔇 [silence/no speech]\n")
        
        except Exception as e:
            print(f"\n❌ Transcription error: {e}\n")
        finally:
            try:
                os.unlink(temp_filename)
            except:
                pass
    
    def run(self) -> None:
        print("\n" + "="*70)
        print("🎤 LIVE TRANSCRIBER - Google Speech-to-Text")
        print("="*70)
        print("Press CTRL+C to stop\n")
        
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
            print("\n" + "="*70)
            print("🛑 STOPPED")
            print("="*70 + "\n")
        except Exception as e:
            print(f"\n❌ Audio error: {e}\n")


def main() -> None:
    transcriber = LiveTranscriber()
    transcriber.run()


if __name__ == "__main__":
    main()