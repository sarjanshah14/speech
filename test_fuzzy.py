#!/usr/bin/env python3
"""
Test script for fuzzy matching changes in deep.py
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from deep import ProductCodeTranscriber

def test_fuzzy_matching():
    """Test the fuzzy matching logic with short candidates."""

    # Mock product codes including short ones
    test_pcodes = [
        "AB", "CD", "EF", "GH", "IJ", "KL", "MN", "OP", "QR", "ST",
        "ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "STU", "VWX", "YZ",
        "ABC123", "DEF456", "GHI789"
    ]

    # Create transcriber with mock data
    transcriber = ProductCodeTranscriber(deepgram_api_key="dummy_key", pcode_list=test_pcodes)

    # Test cases for fuzzy matching (note: matches are returned in lowercase)
    test_cases = [
        # Short candidates (2 chars) - should now be considered for fuzzy matching
        ("AB", "ab", 1.0),  # Exact match
        ("AC", "abc", 0.8),  # Close match to "abc"
        ("CD", "cd", 1.0),  # Exact match
        ("CE", "cd", 0.5),  # Close match

        # Longer candidates
        ("ABC", "abc", 1.0),  # Exact match
        ("ABD", "ab", 0.9),  # Close match to "ab"
        ("XYZ", "yz", 0.9),  # Substring match

        # Edge cases
        ("A", "ab", 0.9),  # Single char matches "ab" with high score due to substring
        ("", "", 0.0),    # Empty, should not match
    ]

    print("Testing fuzzy matching with len(partial_clean) >= 2...")
    print("=" * 60)

    all_passed = True

    for candidate, expected_match, min_score in test_cases:
        actual_match, actual_score = transcriber.find_closest_pcode(candidate, threshold=0.0)  # Low threshold to see all matches

        # Check if we got the expected match with sufficient score
        if expected_match:
            if actual_match == expected_match and actual_score >= min_score:
                status = "✓ PASS"
            else:
                status = f"✗ FAIL (got {actual_match}@{actual_score:.2f}, expected {expected_match}@{min_score})"
                all_passed = False
        else:
            if not actual_match:
                status = "✓ PASS"
            else:
                status = f"✗ FAIL (got {actual_match}@{actual_score:.2f}, expected no match)"
                all_passed = False

        print(f"  {candidate:8} -> {status}")

    print("=" * 60)
    if all_passed:
        print("✓ All fuzzy matching tests passed!")
    else:
        print("✗ Some tests failed!")

    return all_passed

def test_extract_pcode_fuzzy():
    """Test the extract_pcode_and_remaining_text method with fuzzy matching."""

    test_pcodes = ["AB", "CD", "ABC", "DEF", "ABC123"]
    transcriber = ProductCodeTranscriber(deepgram_api_key="dummy_key", pcode_list=test_pcodes)

    # Test transcripts that should trigger fuzzy matching
    test_transcripts = [
        "A B C one two three",  # Should find "ABC" exactly
        "A C C one two three",  # Should find "ABC" fuzzily
        "A B one two three",    # Should find "AB" exactly
        "A C one two three",    # Should find "AB" fuzzily
    ]

    print("\nTesting extract_pcode_and_remaining_text with fuzzy matching...")
    print("=" * 60)

    for transcript in test_transcripts:
        pcode, remaining, confidence = transcriber.extract_pcode_and_remaining_text(transcript)
        print(f"  '{transcript}' -> pcode: '{pcode}', remaining: '{remaining}', confidence: {confidence:.2f}")

if __name__ == "__main__":
    try:
        fuzzy_passed = test_fuzzy_matching()
        test_extract_pcode_fuzzy()

        if fuzzy_passed:
            print("\n✓ Fuzzy matching change validated successfully!")
        else:
            print("\n✗ Fuzzy matching tests failed - please review the implementation!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
