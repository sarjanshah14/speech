# Live Speech Transcriber

This project includes a live speech transcriber script (`live_deep.py`) that uses the Deepgram API for real-time speech-to-text transcription. It continuously listens to audio input, detects speech segments based on silence thresholds, transcribes the audio, and parses the transcript for product codes and quantities using a predefined product list.

## What It Does

The `live_deep.py` script enables real-time speech-to-text transcription of audio input from a microphone. It processes spoken words into text transcripts and attempts to extract product codes and quantities by matching against a list of products stored in an Excel file (`productlist.xlsx`). This is particularly useful for applications like inventory management, where users can speak product names and quantities hands-free, and the system recognizes and logs them automatically.

## How It Works

1. **Audio Capture**: The script uses the `sounddevice` library to continuously record audio from the microphone at a sample rate of 16,000 Hz in mono channel.
2. **Silence Detection**: Audio is buffered in chunks. Silence is detected based on RMS amplitude thresholds (below 300). After 1.5 seconds of silence following speech, the buffered audio is processed.
3. **Transcription**: The buffered audio is saved as a temporary WAV file and sent to the Deepgram API for transcription. The API returns a JSON response with the transcript text.
4. **Parsing**: The transcript is parsed using fuzzy matching logic from `deep.py` to identify product codes (exact matches with high confidence) and quantities (word-to-number conversion).
5. **Output**: Results are printed to the console, including live transcripts, recognized products/quantities, or error messages. The process repeats until stopped.

## Why This Design

- **Real-Time Efficiency**: By detecting silence, the script avoids unnecessary API calls during continuous speech, reducing costs and latency.
- **Hands-Free Operation**: Ideal for scenarios where manual input is impractical, such as in warehouses or during multitasking.
- **Integration with Existing Logic**: Reuses parsing logic from `deep.py` for consistency with offline transcription features.
- **Error Handling**: Includes robust error handling for transcription failures, API issues, and audio problems, ensuring reliability.
- **Customization**: Parameters like silence thresholds and sample rates are configurable, allowing adaptation to different environments or audio qualities.

## Features

- **Real-time Audio Recording**: Captures audio from the microphone in chunks.
- **Silence Detection**: Processes audio only after detecting silence to avoid constant transcription.
- **Speech Transcription**: Uses Deepgram's API to convert speech to text.
- **Product Parsing**: Extracts product codes and quantities from transcripts using fuzzy matching against a product list in `productlist.xlsx`.
- **Output**: Displays live transcripts, recognized products, and quantities in the console.

## Prerequisites

- Python 3.7+
- A Deepgram API key (set via environment variable or file)
- Microphone access for audio input
- `productlist.xlsx` file in the project root (contains product codes for matching)

## Installation

1. Clone or download the project files.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure `productlist.xlsx` is present in the project directory.

## Usage

### Running the Live Transcriber

To start the live transcriber:

```bash
python live_deep.py
```

### API Key Configuration

The script looks for the Deepgram API key in the following order:
1. Environment variable: `DEEPGRAM_API_KEY`
2. File: `keys/deepgram.key` (default path)

If neither is found, the script will exit with an error message.

You can set the API key via environment variable:
```bash
export DEEPGRAM_API_KEY=your_api_key_here
python live_deep.py
```

Or place the key in `keys/deepgram.key`.

### Audio Settings

- **Sample Rate**: 16,000 Hz
- **Channels**: 1 (mono)
- **Chunk Duration**: 2 seconds
- **Silence Threshold**: 1.5 seconds of silence triggers processing
- **Silence Amplitude**: RMS below 300 is considered silence

### Output Examples

- Live transcript: `🎤 Live: hello world`
- Recognized product: `✅ Product: ABC123 | Qty: 5`
- Errors: `❌ Can't recognize product code` or `❌ Transcription error: ...`

Press `CTRL+C` to stop the transcriber.

## Dependencies

The project relies on the following key libraries (see `requirements.txt` for full list):
- `deepgram-sdk`: For speech-to-text API
- `numpy`: Audio data processing
- `sounddevice`: Audio input handling
- `openpyxl`: Reading Excel product list
- Custom modules: `deep.py` for product parsing logic

## Project Structure

- `live_deep.py`: Main live transcriber script
- `deep.py`: Offline transcription and product parsing logic
- `productlist.xlsx`: Excel file with product codes
- `keys/deepgram.key`: API key file (optional)
- `requirements.txt`: Python dependencies
- Other files: Additional scripts like `speech.py`, `multi.py`, etc.

## Troubleshooting

- **No audio input**: Ensure microphone permissions and that `sounddevice` can access the device.
- **API errors**: Check your Deepgram API key and internet connection.
- **Product not recognized**: Verify `productlist.xlsx` format and content.
- **Silence detection issues**: Adjust `SILENCE_SECONDS` or `SILENCE_AMPLITUDE` in the code if needed.

## License

[Add license information if applicable]
