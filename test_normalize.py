from deep import ProductCodeTranscriber

# Create a dummy transcriber instance (we don't need API key for testing normalize_pcode_portion)
transcriber = ProductCodeTranscriber.__new__(ProductCodeTranscriber)

# Test cases for normalize_pcode_portion
test_cases = [
    # Letter mapping tests
    ("zed", "z"),
    ("bee", "b"),
    ("see", "c"),
    ("dee", "d"),
    ("pee", "p"),
    ("tee", "t"),
    ("vee", "v"),
    ("doubleyou", "w"),
    ("ex", "x"),
    ("why", "y"),
    ("jay", "j"),
    ("kay", "k"),
    ("ell", "l"),
    ("em", "m"),
    ("en", "n"),
    ("oh", "o"),
    ("queue", "q"),
    ("are", "r"),
    ("ess", "s"),
    ("you", "u"),

    # Compound number patterns in product codes
    ("one hundred eight", "108"),
    ("two hundred thirty five", "235"),
    ("one hundred eleven", "111"),
    ("three hundred twenty", "320"),
    ("four hundred", "400"),

    # Mixed cases
    ("h one hundred eight", "h108"),
    ("l two hundred fifty seven", "l257"),
    ("a one hundred", "a100"),

    # Ensure multipliers are removed after processing
    ("h one hundred eight thousand", "h108"),
    ("l two lakh three", "l23"),

    # Digit-by-digit cases
    ("one two three", "123"),
    ("five six seven", "567"),

    # Tens corrections
    ("fifty seven", "5 7"),
    ("twenty three", "2 3"),
    ("thirty", "3 0"),

    # Special patterns
    ("triple zero", "000"),
    ("double five", "55"),
]

print("Testing normalize_pcode_portion method...")
print("=" * 50)

all_passed = True
for input_text, expected in test_cases:
    result = transcriber.normalize_pcode_portion(input_text)
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        all_passed = False
    print(f"{status}: '{input_text}' -> '{result}' (expected: '{expected}')")

print("=" * 50)
if all_passed:
    print("All tests PASSED!")
else:
    print("Some tests FAILED!")
