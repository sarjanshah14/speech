import requests
import json
import re
from typing import Dict, List, Tuple
from pathlib import Path
import openpyxl
from difflib import SequenceMatcher

class ProductCodeTranscriber:
    def __init__(self, deepgram_api_key: str, pcode_list: List[str] = None, excel_file: str = "productlist.xlsx"):
        """
        Initialize the transcriber with Deepgram API key and product codes.
        """
        self.api_key = deepgram_api_key
        self.base_url = "https://api.deepgram.com/v1/listen"
        
        if pcode_list is None:
            self.pcode_list = self.load_pcodes_from_excel(excel_file)
        else:
            self.pcode_list = pcode_list
        
        print(f"Loaded {len(self.pcode_list)} product codes")
        
        self.headers = {
            "Authorization": f"Token {deepgram_api_key}"
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
            print(f"Error loading Excel file: {e}")
            return []
    
    def transcribe_audio(self, audio_file_path: str) -> Dict:
        """Transcribe audio file using Deepgram API."""
        params = {
            "model": "nova-2",
            "language": "en",
            "punctuate": "true",
            "paragraphs": "true"
        }
        
        with open(audio_file_path, "rb") as audio_file:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                params=params,
                data=audio_file
            )
        
        if response.status_code != 200:
            raise Exception(f"Deepgram API error: {response.status_code} - {response.text}")
        
        return response.json()
    
    def words_to_number(self, text: str) -> int:
        """Convert written number words to integer (for quantity parsing).
        Supports: hundred, thousand, lakh/lac, crore, million, billion, trillion
        Properly handles cases like "one thirty five" = 135 and "two sixty seven" = 267
        """
        text = text.lower().strip()
        
        # Remove common filler words
        text = re.sub(r'\band\b', ' ', text)
        
        word_values = {
            'zero': 0, 'oh': 0,
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19,
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        
        # Multipliers in ascending order for proper handling
        multipliers = {
            'hundred': 100,
            'thousand': 1000,
            'lakh': 100000,
            'lac': 100000,
            'lakhs': 100000,
            'lacs': 100000,
            'crore': 10000000,
            'crores': 10000000,
            'million': 1000000,
            'millions': 1000000,
            'billion': 1000000000,
            'billions': 1000000000,
            'trillion': 1000000000000,
            'trillions': 1000000000000
        }
        
        words = text.split()
        result = 0
        current = 0
        
        i = 0
        while i < len(words):
            word = words[i].strip()
            if not word:
                i += 1
                continue
            
            if word in word_values:
                # Check if this is part of a larger number or standalone
                next_word = words[i + 1] if i + 1 < len(words) else None
                
                # If next word is a tens/compound (thirty, forty, etc.), this forms a compound number
                # "one thirty" means 130, "two sixty" means 260
                if next_word in ['twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']:
                    # Multiply current digit by 100 and add the tens value
                    current += word_values[word] * 100
                    i += 1  # Move to tens word
                    current += word_values[words[i]]
                    # Check for ones digit after tens
                    if i + 1 < len(words) and words[i + 1] in word_values and word_values[words[i + 1]] < 10:
                        i += 1
                        current += word_values[words[i]]
                else:
                    current += word_values[word]
            elif word in multipliers:
                multiplier_value = multipliers[word]
                
                # Handle 'hundred' - it multiplies the current value
                if word == 'hundred':
                    if current == 0:
                        current = 1
                    current *= multiplier_value
                else:
                    # For larger multipliers (thousand, lakh, crore, etc.)
                    if current == 0:
                        current = 1
                    result += current * multiplier_value
                    current = 0
            
            i += 1
        
        # Add any remaining current value
        result += current
        return result
    
    def normalize_pcode_portion(self, text: str) -> str:
        """Normalize only the product code portion (letters + numbers).
        - Converts letter names (e.g., "zed") to letters
        - Expands double/triple digits
        - Converts number words (teens, tens, "X hundred Y") to digits
        - Removes large multipliers like thousand/crore that should not appear in pcodes
        """
        text = text.lower().strip()

        # Basic cleanup: remove filler 'and' and punctuation
        text = re.sub(r'\band\b', ' ', text)
        text = re.sub(r'[.,!?;:]', ' ', text)

        # Map letter names commonly misrecognized
        letter_name_map = {
            'zed': 'z',
            'zee': 'z',
        }
        for name, letter in letter_name_map.items():
            text = re.sub(rf'\b{name}\b', letter, text)

        # Special case: "double u" -> 'w'
        text = re.sub(r'\bdouble\s+u\b', 'w', text)

        # Expand double/triple digit sequences first
        special_patterns = [
            (r'\btriple\s+zero\b', '000'), (r'\btriple\s+one\b', '111'), (r'\btriple\s+two\b', '222'),
            (r'\btriple\s+three\b', '333'), (r'\btriple\s+four\b', '444'), (r'\btriple\s+five\b', '555'),
            (r'\btriple\s+six\b', '666'), (r'\btriple\s+seven\b', '777'), (r'\btriple\s+eight\b', '888'),
            (r'\btriple\s+nine\b', '999'),
            (r'\bdouble\s+zero\b', '00'), (r'\bdouble\s+one\b', '11'), (r'\bdouble\s+two\b', '22'),
            (r'\bdouble\s+three\b', '33'), (r'\bdouble\s+four\b', '44'), (r'\bdouble\s+five\b', '55'),
            (r'\bdouble\s+six\b', '66'), (r'\bdouble\s+seven\b', '77'), (r'\bdouble\s+eight\b', '88'),
            (r'\bdouble\s+nine\b', '99'),
        ]
        for pattern, replacement in special_patterns:
            text = re.sub(pattern, replacement, text)

        # Convert "X hundred Y" constructs BEFORE removing 'hundred'
        ones = ['zero','one','two','three','four','five','six','seven','eight','nine']
        tens_words = ['', 'ten', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        teens = ['ten','eleven','twelve','thirteen','fourteen','fifteen','sixteen','seventeen','eighteen','nineteen']

        for hundreds_digit in range(1, 10):
            hundreds_word = ones[hundreds_digit]
            # ones after hundred: one hundred eight -> 108
            for ones_digit in range(0, 10):
                ones_word = ones[ones_digit]
                pattern = rf'\b{hundreds_word}\s+hundred\s+{ones_word}\b'
                replacement = f'{hundreds_digit}0{ones_digit}'
                text = re.sub(pattern, replacement, text)
            # teens after hundred: one hundred fourteen -> 114
            for teen_idx, teen_word in enumerate(teens):
                pattern = rf'\b{hundreds_word}\s+hundred\s+{teen_word}\b'
                replacement = f'{hundreds_digit}1{teen_idx}'
                text = re.sub(pattern, replacement, text)
            # tens (+ optional ones): one hundred thirty two -> 132; one hundred thirty -> 130
            for tens_digit in range(2, 10):
                tens_word = tens_words[tens_digit]
                # tens + ones
                for ones_digit in range(1, 10):
                    ones_word = ones[ones_digit]
                    pattern = rf'\b{hundreds_word}\s+hundred\s+{tens_word}\s+{ones_word}\b'
                    replacement = f'{hundreds_digit}{tens_digit}{ones_digit}'
                    text = re.sub(pattern, replacement, text)
                # tens only
                pattern = rf'\b{hundreds_word}\s+hundred\s+{tens_word}\b'
                replacement = f'{hundreds_digit}{tens_digit}0'
                text = re.sub(pattern, replacement, text)
            # X hundred -> X00
            pattern = rf'\b{hundreds_word}\s+hundred\b'
            replacement = f'{hundreds_digit}00'
            text = re.sub(pattern, replacement, text)

        # Convert teens standalone to 2-digit numbers: sixteen -> 16
        teen_map = {
            'ten':'10','eleven':'11','twelve':'12','thirteen':'13','fourteen':'14','fifteen':'15','sixteen':'16','seventeen':'17','eighteen':'18','nineteen':'19'
        }
        for w, d in teen_map.items():
            text = re.sub(rf'\b{w}\b', d, text)

        # Convert tens + ones pairs into 2-digit numbers: thirty two -> 32; tens alone -> 30
        def tens_to_number(m):
            tens_word = m.group(1)
            one_word = m.group(2)
            tens_val = {
                'twenty':'20','thirty':'30','forty':'40','fifty':'50','sixty':'60','seventy':'70','eighty':'80','ninety':'90'
            }[tens_word]
            ones_val = {
                'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9'
            }[one_word]
            return tens_val[0] + ones_val

        text = re.sub(r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\s+(one|two|three|four|five|six|seven|eight|nine)\b', tens_to_number, text)
        text = re.sub(r'\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)\b',
                      lambda m: {'twenty':'20','thirty':'30','forty':'40','fifty':'50','sixty':'60','seventy':'70','eighty':'80','ninety':'90'}[m.group(1)],
                      text)

        # Remove large multipliers not intended for pcodes (keep 'hundred' already handled)
        text = re.sub(r'\b(thousand|lakh|lakhs|lac|lacs|crore|crores|million|millions|billion|billions|trillion|trillions)\b', ' ', text)

        # Convert remaining single-digit words
        simple_digit_map = {
            'zero': '0', 'oh': '0',
            'one': '1', 'two': '2', 'to': '2', 'too': '2',
            'three': '3', 'four': '4', 'for': '4',
            'five': '5', 'six': '6', 'seven': '7',
            'eight': '8', 'ate': '8', 'nine': '9'
        }
        for word, digit in simple_digit_map.items():
            text = re.sub(r'\b' + word + r'\b', digit, text)

        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def find_closest_pcode(self, candidate: str, threshold: float = 0.8) -> Tuple[str, float]:
        """
        Find the closest matching product code from the list.
        Returns: (best_match, similarity_score)
        """
        if not candidate:
            return "", 0.0
        
        candidate_lower = candidate.lower()
        best_match = ""
        best_score = 0.0
        
        for pcode in self.pcode_list:
            pcode_clean = str(pcode).lower()
            
            # Exact match
            if pcode_clean == candidate_lower:
                return pcode_clean, 1.0
            
            # Calculate similarity
            similarity = SequenceMatcher(None, candidate_lower, pcode_clean).ratio()
            
            # Also check if candidate is a substring or vice versa
            if candidate_lower in pcode_clean or pcode_clean in candidate_lower:
                # Boost score for substring matches
                substring_score = max(len(candidate_lower), len(pcode_clean)) / max(len(candidate_lower), len(pcode_clean))
                similarity = max(similarity, substring_score * 0.9)
            
            if similarity > best_score:
                best_score = similarity
                best_match = pcode_clean
        
        # Only return if above threshold
        if best_score >= threshold:
            return best_match, best_score
        
        return "", 0.0
    
    def extract_pcode_and_remaining_text(self, text: str) -> Tuple[str, str, float]:
        """
        Extract pcode and return the remaining text after the pcode.
        Returns: (pcode, remaining_text, confidence_score)
        """
        text_lower = text.lower().strip()
        # Normalize punctuation early so tokens like 'p,' or 'five.' don't break parsing
        text_lower = re.sub(r'[.,!?;:]', ' ', text_lower)
        words = text_lower.split()
        
        if not words:
            return "", "", 0.0
        
        # Check if first word is misrecognized "eight" -> "h"
        first_word = words[0]
        if first_word == 'eight' and len(words) > 1:
            next_word = words[1]
            digit_words = {'zero', 'oh', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                          'double', 'triple'}
            if next_word in digit_words or next_word.startswith('doub') or next_word.startswith('trip'):
                text_lower = 'h ' + ' '.join(words[1:])
                words = text_lower.split()
        
        # Check if transcript is missing the first letter - try all single letters
        first_word_is_digit = words[0] in {'zero', 'oh', 'one', 'two', 'three', 'four', 'five', 
                                            'six', 'seven', 'eight', 'nine', 'double', 'triple'} or \
                              words[0].startswith('doub') or words[0].startswith('trip') or \
                              words[0].isdigit()
        
        if first_word_is_digit:
            # Try prefixing with each letter a-z
            for letter in 'abcdefghijklmnopqrstuvwxyz':
                test_text = letter + ' ' + text_lower
                test_words = test_text.split()
                
                # Try to match with this letter prefix
                for end_idx in range(1, min(len(test_words) + 1, 10)):
                    partial_text = ' '.join(test_words[:end_idx])
                    normalized_partial = self.normalize_pcode_portion(partial_text)
                    normalized_clean = normalized_partial.replace(' ', '')
                    match = re.match(r'^([a-z]+\d+)', normalized_clean)
                    
                    if match:
                        candidate = match.group(1)
                        for pcode in self.pcode_list:
                            pcode_clean = str(pcode).lower()
                            if pcode_clean == candidate:
                                # Found a match! Return it
                                remaining_text = ' '.join(words[end_idx-1:])
                                return pcode_clean, remaining_text, 1.0
        
        # Find where the pcode ends by matching against known pcodes
        # Try progressively longer sequences and check each one
        best_match = ""
        best_match_end_index = 0
        best_confidence = 0.0
        
        for end_idx in range(1, min(len(words) + 1, 20)):
            partial_text = ' '.join(words[:end_idx])
            normalized_partial = self.normalize_pcode_portion(partial_text)
            
            # Extract letter-digit sequence
            normalized_clean = normalized_partial.replace(' ', '')
            
            # DEBUG: Show what we're processing
            print(f"  DEBUG: Processing words {words[:end_idx]} -> normalized: '{normalized_clean}'")
            
            # Look for letter-digit patterns of various lengths
            # Try different patterns to capture the full product code
            patterns = [
                r'^([a-z]+\d+)',  # Original pattern
                r'^([a-z]+\d[\d]*)',  # More flexible digit matching
            ]
            
            candidate = ""
            for pattern in patterns:
                match = re.match(pattern, normalized_clean)
                if match:
                    candidate = match.group(1)
                    break
            
            if candidate:
                # Debug: print what we're checking
                print(f"  DEBUG: Checking candidate '{candidate}' from normalized: '{normalized_clean}'")
                
                # Check if this EXACT candidate is in our list
                if candidate in [str(p).lower() for p in self.pcode_list]:
                    print(f"  DEBUG: Found exact match '{candidate}'!")
                    best_match = candidate
                    best_match_end_index = end_idx
                    best_confidence = 1.0
                    # STOP HERE - don't look for longer matches
                    break
                
                # Also check if we have a partial match that could be extended
                for pcode in self.pcode_list:
                    pcode_clean = str(pcode).lower()
                    if pcode_clean.startswith(candidate):
                        print(f"  DEBUG: Found partial match '{candidate}' -> '{pcode_clean}'")
                        continue
        
        # If exact match found, return it
        if best_match:
            remaining_text = ' '.join(words[best_match_end_index:])
            return best_match, remaining_text, best_confidence
        
        # No exact match - try fuzzy matching with extended search
        for end_idx in range(1, min(len(words) + 1, 12)):
            partial_text = ' '.join(words[:end_idx])
            normalized_partial = self.normalize_pcode_portion(partial_text)
            normalized_clean = normalized_partial.replace(' ', '')
            
            # Try multiple patterns for fuzzy matching
            patterns = [
                r'^([a-z]+\d+)',
                r'^([a-z]+\d[\d]*)',
            ]
            
            candidate = ""
            for pattern in patterns:
                match = re.match(pattern, normalized_clean)
                if match:
                    candidate = match.group(1)
                    break
            
            if candidate:
                closest_match, score = self.find_closest_pcode(candidate, threshold=0.7)
                if closest_match and score > best_confidence:
                    best_match = closest_match
                    best_match_end_index = end_idx
                    best_confidence = score
                    print(f"  DEBUG: Fuzzy match '{candidate}' -> '{closest_match}' (score: {score})")
        
        if best_match:
            remaining_text = ' '.join(words[best_match_end_index:])
            return best_match, remaining_text, best_confidence
        
        return "", text_lower, 0.0
    
    def parse_quantity(self, text: str) -> str:
        """
        Parse quantity from text. Handles both:
        1. Digit-by-digit speech: "three two one" -> "321"
        2. Indian style: "three seventy two" -> "372" (not 75!)
        3. Natural numbers: "three hundred twenty one" -> "321"
        """
        text = text.lower().strip()
        
        if not text:
            return ""
        
        # Remove ALL punctuation first
        text = re.sub(r'[.,!?;:]', '', text)
        
        # Remove common words like quantity units
        text = re.sub(r'\b(pieces|piece|units|unit|items|item|species|pcs|p\s*c|pc|reports|report)\b', '', text, flags=re.IGNORECASE)
        text = text.strip()
        
        if not text:
            return ""
        
        # Split into words and filter out empty strings
        words = [w for w in text.split() if w]
        
        if not words:
            return ""
        
        # Define word categories
        single_digit_words = {'zero', 'oh', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'}
        multiplier_words = {'hundred', 'thousand', 'lakh', 'lakhs', 'lac', 'lacs', 'crore', 'crores', 
                           'million', 'millions', 'billion', 'billions', 'trillion', 'trillions', 'and'}
        compound_words = {'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 
                         'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty', 
                         'sixty', 'seventy', 'eighty', 'ninety'}
        
        # Check if we have multipliers (hundred, thousand, etc.) - indicates natural number
        has_multiplier = any(word in multiplier_words for word in words)
        
        # If there are multipliers, use natural number parsing
        if has_multiplier:
            try:
                qty_num = self.words_to_number(' '.join(words))
                print(f"  DEBUG QTY: Natural number mode (has multipliers): {words} -> {qty_num}")
                return str(qty_num)
            except Exception as e:
                print(f"  DEBUG QTY: Failed natural number parsing: {words}: {e}")
                return ""
        
        # No multipliers - could be digit-by-digit OR compound
        # Check if ALL words are single digits
        all_single_digits = all(word in single_digit_words for word in words)
        
        if all_single_digits:
            # Pure digit-by-digit: "three two one" -> "321"
            digit_map = {
                'zero': '0', 'oh': '0', 'one': '1', 'two': '2', 'three': '3', 
                'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
            }
            result = ''.join(digit_map.get(word, '') for word in words)
            print(f"  DEBUG QTY: Digit-by-digit mode: {words} -> '{result}'")
            return result if result else ""
        
        # Has compound words but no multipliers - ALWAYS use natural number arithmetic
        try:
            qty_num = self.words_to_number(' '.join(words))
            print(f"  DEBUG QTY: Natural number with compounds: {words} -> {qty_num}")
            return str(qty_num)
        except Exception as e:
            print(f"  DEBUG QTY: Failed compound parsing: {words}: {e}")
            return ""
    
    def extract_pcode_and_qty(self, transcript: str) -> Tuple[str, str, float]:
        """
        Extract both pcode and quantity from transcript.
        First finds the exact pcode, then parses remaining text as quantity.
        Returns: (pcode, quantity, confidence_score)
        """
        # Extract pcode and get remaining text
        pcode, remaining_text, confidence = self.extract_pcode_and_remaining_text(transcript)
        
        if not pcode:
            return "", "", 0.0
        
        # Parse remaining text as quantity
        qty_str = self.parse_quantity(remaining_text)
        
        return pcode, qty_str, confidence
    
    def process_audio(self, audio_file_path: str) -> Dict:
        """Complete pipeline: transcribe -> extract pcode -> separate quantity."""
        print(f"Transcribing: {audio_file_path}")
        transcription_response = self.transcribe_audio(audio_file_path)
        
        transcript = transcription_response["results"]["channels"][0]["alternatives"][0]["transcript"]
        print(f"Transcribed text: {transcript}")
        
        pcode, qty, confidence = self.extract_pcode_and_qty(transcript)
        
        confidence_label = "EXACT" if confidence == 1.0 else f"FUZZY ({confidence:.0%})"
        print(f"Extracted pcode: {pcode} [{confidence_label}], qty: {qty}")
        
        return {
            "audio_file": audio_file_path,
            "transcript": transcript,
            "pcode": pcode,
            "quantity": qty,
            "confidence": confidence,
            "full_response": transcription_response
        }
    
    def batch_process(self, audio_files: List[str]) -> List[Dict]:
        """Process multiple audio files."""
        results = []
        for audio_file in audio_files:
            try:
                result = self.process_audio(audio_file)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                results.append({
                    "audio_file": audio_file,
                    "error": str(e)
                })
        
        return results


if __name__ == "__main__":
    with open("keys/deepgram.key", "r") as f:
        api_key = f.read().strip()
    
    transcriber = ProductCodeTranscriber(api_key)
    
    # Get all .mp3 files from audio folder
    audio_folder = Path("audio")
    if not audio_folder.exists():
        print(f"Error: '{audio_folder}' folder not found!")
        exit(1)
    
    audio_files = list(audio_folder.glob("*.mp3"))
    
    if not audio_files:
        print(f"No .mp3 files found in '{audio_folder}' folder!")
        exit(1)
    
    print(f"\nFound {len(audio_files)} .mp3 files to process\n")
    print("=" * 80)
    
    # Batch process all files
    results = transcriber.batch_process([str(f) for f in audio_files])
    
    # Print summary results
    print("\n" + "=" * 80)
    print("\n=== SUMMARY RESULTS ===\n")
    
    for r in results:
        if 'error' in r:
            print(f"❌ {r['audio_file']}: ERROR - {r['error']}")
        else:
            confidence = r.get('confidence', 0.0)
            confidence_icon = "✓" if confidence == 1.0 else "⚠"
            confidence_label = "EXACT" if confidence == 1.0 else f"FUZZY ({confidence:.0%})"
            
            print(f"{confidence_icon} {Path(r['audio_file']).name} [{confidence_label}]")
            print(f"  Transcript: {r['transcript']}")
            print(f"  Product Code: {r['pcode']}")
            print(f"  Quantity: {r['quantity']}")
            print()
