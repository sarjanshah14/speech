# Layered Speech-to-Text Architecture Summary

## Overview

This document explains the layered architecture of the refactored Google Live Speech-to-Text Transcriber (`test.py`). The system processes speech input through four distinct stages, each with specific responsibilities and tuning capabilities.

## Architecture Diagram

```
Raw Speech Input
       ↓
┌─────────────────┐
│   Stage 1:      │
│ Raw Cleanup     │
│ Layer           │
└─────────────────┘
       ↓
┌─────────────────┐
│   Stage 2:      │
│ Linguistic      │
│ Normalization   │
└─────────────────┘
       ↓
┌─────────────────┐
│   Stage 3:      │
│ Pattern         │
│ Extraction      │
└─────────────────┘
       ↓
┌─────────────────┐
│   Stage 4:      │
│ Confidence &    │
│ Recovery        │
└─────────────────┘
       ↓
Product Code + Quantity + Confidence Score
```

## Stage 1: Raw Cleanup Layer

### Purpose
Basic text sanitation and noise removal from speech transcription.

### Components
- **Class**: `RawCleanupLayer`
- **Input**: Raw transcribed text from Google Speech API
- **Output**: Cleaned text ready for linguistic processing

### Processing Steps

1. **Text Normalization**
   ```python
   text = text.lower().strip()
   ```

2. **Punctuation Removal**
   ```python
   # Remove unwanted punctuation (keep dots and spaces)
   text = re.sub(r'[,!?;:\'"()]', '', text)
   ```

3. **Whitespace Cleanup**
   ```python
   text = ' '.join(text.split())  # Remove extra whitespace
   ```

4. **Common Transcription Fixes**
   ```python
   text = re.sub(r'\bp\s+c\b', 'pieces', text)
   text = re.sub(r'\bpc\b', 'pieces', text)
   text = re.sub(r'\bzed\b', 'z', text)
   text = re.sub(r'\band\b', ' ', text)
   ```

5. **Misheard Word Corrections**
   ```python
   text = re.sub(r'\bcome\b', 'cum', text)
   text = re.sub(r'\bpace\b', 'piece', text)
   text = re.sub(r'\bpage\b', 'piece', text)
   ```

### Tuning Parameters

- **Punctuation Patterns**: Modify `r'[,!?;:\'"()]'` to include/exclude specific punctuation
- **Transcription Fixes**: Add new common mishearings to the replacement dictionary
- **Whitespace Handling**: Adjust spacing rules for different languages or accents

### Debug Output
```
[Stage 1: Raw Cleanup] Input: 'come p c double 2 3'
[Stage 1: Raw Cleanup] Output: 'cum pieces double 2 3'
```

## Stage 2: Linguistic Normalization

### Purpose
Convert spoken language patterns into standardized text representations suitable for product code matching.

### Components
- **Class**: `LinguisticNormalizer`
- **Input**: Cleaned text from Stage 1
- **Output**: Normalized text with numbers, dots, and patterns converted

### Processing Steps

#### 2.1 Dot Pattern Handling
```python
# Handle "dot" patterns BEFORE number conversions
text = re.sub(r'\bdots?\s+dots?\b', '..', text)  # "dots dots" → ".."
text = re.sub(r'\bdots?\b', '.', text)           # "dot" → "."
```

**Example**: `"s dot ss dot de"` → `"s.ss.de"`

#### 2.2 Double/Triple Pattern Processing
```python
# Handle "double 2 3" → "223"
text = re.sub(r'\bdouble\s+(\d)\s+(\d)\b', r'\1\1\2', text)

# Handle "z triple 0 3" → "z0003"
text = re.sub(r'([a-z]+)\s+triple\s+(\d)\s+(\d)', r'\1\2\2\2\3', text)
```

**Key Fixes**:
- ✅ `"double 2 3"` → `"223"` (not `"22 3"`)
- ✅ `"z triple 0 3"` → `"z0003"` (not `"z03"`)

#### 2.3 Number Word Conversion
```python
number_words = {
    'zero': '0', 'oh': '0', 'one': '1', 'two': '2', 'three': '3',
    'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9'
}
```

**Protected Patterns**: Prevents `"3 three"` from becoming `"33"` (quantity errors)

#### 2.4 Compound Number Handling
```python
compound_patterns = [
    (r'\btwenty\s+one\b', '21'), (r'\btwenty\s+two\b', '22'),
    # ... up to ninety-nine
]
```

#### 2.5 Special Character Fixes
```python
# "eight" at start might be "H" (only if followed by number word)
if words[0] == 'eight' and words[1] in digit_words:
    text = 'h ' + ' '.join(words[1:])
```

### Tuning Parameters

- **Number Word Dictionary**: Add new number words or regional variations
- **Compound Patterns**: Extend for higher numbers (hundred, thousand, etc.)
- **Double/Triple Rules**: Modify patterns for different speech patterns
- **Dot Handling**: Adjust for different separator conventions

### Debug Output
```
[NORM] After dot handling: 's . ss . de'
[NORM] After double/triple: '223'
[NORM] After compound numbers: '223'
[NORM] After number word conversion: '223'
[NORM] After special fixes: '223'
```

## Stage 3: Pattern Extraction

### Purpose
Extract all possible product code candidates from normalized text.

### Components
- **Class**: `PatternExtractor`
- **Input**: Normalized text from Stage 2
- **Output**: List of candidate product codes

### Processing Steps

#### 3.1 Spaced Number Extraction
```python
# Extract "35 335" → "35335"
spaced_numbers = re.findall(r'\d+(?:\s+\d+)+', clean_text)
for match in spaced_numbers:
    compact = match.replace(' ', '')
    if len(compact) >= 4:
        candidates.append(compact)
```

#### 3.2 Dot-Separated Patterns
```python
# Codes with dots: S1R.SS.E, JC.TM.NPP
dot_patterns = re.findall(r'[a-z]+\d*[a-z]*\.[a-z0-9.]+', no_space)
```

#### 3.3 Letter-Digit Combinations
```python
# Letter(s) + digit(s): INC35335, z0003
matches = re.findall(r'[a-z]{1,15}\d+', no_space)
```

#### 3.4 Pure Numeric Codes
```python
# Pure numeric codes (3+ digits)
pure_numeric = re.findall(r'\d{3,}', no_space)
```

#### 3.5 Spaced Letter Patterns
```python
# Letter + space + letter + digits
spaced_matches = re.findall(r'[a-z]\s+[a-z]\d+', clean_text)
```

### Tuning Parameters

- **Minimum Length**: Adjust `len(compact) >= 4` for different code lengths
- **Pattern Complexity**: Modify regex patterns for different code formats
- **Character Limits**: Change `[a-z]{1,15}` for different prefix lengths
- **Numeric Threshold**: Adjust `\d{3,}` for minimum digit requirements

### Debug Output
```
[EXTRACT] Spaced number candidate: '35 335' → '35335'
[Stage 3: Pattern Extraction] Candidates: ['35335', 'inc35335']
```

## Stage 4: Confidence & Recovery

### Purpose
Match candidates against product database with confidence scoring and fallback strategies.

### Components
- **Class**: `Matcher`
- **Input**: List of candidate codes from Stage 3
- **Output**: Best match with confidence score and remainder

### Processing Steps

#### 4.1 Exact Matching
```python
# Check cache first
if candidate_lower in self.match_cache:
    return cached_match, "", 1.0

# Exact match
if candidate_lower == pcode_clean:
    confidence = 1.0
```

#### 4.2 Prefix Matching
```python
# Candidate starts with product code
elif candidate_lower.startswith(pcode_clean):
    remainder = candidate_lower[len(pcode_clean):]
    confidence = 0.9
```

#### 4.3 Fuzzy Matching (Dots Removed)
```python
def _fuzzy_match_with_dots(self, candidate):
    candidate_clean = candidate.replace('.', '').replace(' ', '').lower()
    # Match against product codes with dots removed
```

#### 4.4 Character Confusion Correction
```python
similar_chars = {
    'a': ['e', 'o'], 'b': ['d', 'p', 'v'], 'c': ['s', 'k', 'g'],
    # ... comprehensive mapping
}
```

#### 4.5 Recovery Mode
```python
def _recovery_mode(self, candidates):
    # Try partial matches (at least 3 characters)
    # Use Levenshtein distance for similarity
    similarity = SequenceMatcher(None, candidate_clean, pcode_no_dot).ratio()
    if similarity >= 0.6:
        return str(self.pcode_list[i]), "", similarity
```

### Confidence Scoring

- **1.0**: Exact match (cached or direct)
- **0.9**: Prefix match
- **0.8**: Fuzzy match (dots removed)
- **0.7**: Character confusion correction
- **0.6+**: Recovery mode (similarity-based)

### Tuning Parameters

- **Similarity Threshold**: Adjust `similarity >= 0.6` for recovery mode
- **Character Mappings**: Add new similar-sounding character pairs
- **Cache Size**: Control memory usage vs. performance
- **Confidence Thresholds**: Modify scoring for different match types

### Debug Output
```
[MATCH] Exact: '35335' → '35335'
[MATCH] Prefix: '223' → '2' (remainder: 23)
[MATCH] Fuzzy dot: 'z0003' → '13ED0003'
[MATCH] Similar char: 'inc35335' → 'INC35335'
```

## Error Handling & Recovery

### Unified Error Handler
```python
class ErrorHandler:
    @staticmethod
    def safe_execute(func, stage, default_return=None, context=""):
        try:
            return func()
        except Exception as e:
            ErrorHandler.log_error(stage, e, context)
            return default_return
```

### Graceful Degradation
- Each stage continues processing even if previous stage fails
- Default values prevent crashes
- Detailed error logging for debugging

## Testing & Debugging

### Test Mode
```bash
# Test with sample texts
python test.py --test --debug

# Test specific text
python test.py --text "double 2 3" --debug
```

### Debug Output Levels
- **Stage-by-stage processing**: Shows transformation at each step
- **Candidate extraction**: Lists all possible matches
- **Matching process**: Shows confidence scoring
- **Error details**: Full stack traces when debug enabled

## Performance Optimizations

### Caching
```python
# Cache recent successful matches
self.match_cache: Dict[str, str] = {}
```

### Pattern Pre-compilation
- Regex patterns compiled once during initialization
- Efficient string operations
- Minimal memory allocation

## Extension Points

### Adding New Normalization Rules
1. Add patterns to `LinguisticNormalizer`
2. Update debug output
3. Test with sample inputs

### Adding New Matching Strategies
1. Extend `Matcher` class
2. Add confidence scoring
3. Update recovery mode

### Adding New Error Handling
1. Extend `ErrorHandler` class
2. Add stage-specific error types
3. Update logging format

## Configuration Examples

### For Different Languages
```python
# Add language-specific number words
number_words.update({
    'uno': '1', 'dos': '2', 'tres': '3'  # Spanish
})
```

### For Different Product Code Formats
```python
# Adjust pattern extraction
pure_numeric = re.findall(r'\d{2,}', no_space)  # 2+ digits instead of 3+
```

### For Different Confidence Requirements
```python
# Require higher confidence for production
if confidence < 0.8:
    return "", "", 0.0
```

## Monitoring & Metrics

### Key Metrics to Track
- **Processing Time**: Per stage and total
- **Confidence Distribution**: Match quality over time
- **Error Rates**: By stage and error type
- **Cache Hit Rate**: Performance optimization
- **Recovery Mode Usage**: Fallback effectiveness

### Logging Strategy
- **Production**: Error-only logging
- **Debug**: Full stage-by-stage output
- **Development**: Verbose with timing information

This layered architecture provides a robust, extensible foundation for speech-to-text processing with clear separation of concerns and comprehensive error handling.
