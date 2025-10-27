import os
import sys
import time
import tempfile
import threading
from typing import Optional

import numpy as np
import sounddevice as sd

# Reuse normalization + fuzzy match logic from deep.py
from deep import ProductCodeTranscriber


SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_DURATION = 2.0  # seconds per chunk
SILENCE_SECONDS = 1.5  # silence to trigger processing
SILENCE_AMPLITUDE = 300


class LiveTranscriber:
    def __init__(self, api_key: str, excel_file: str = "productlist.xlsx") -> None:
        self.api_key = api_key
        self.should_stop = False

        # Parsing helper from existing offline pipeline
        self.parser = ProductCodeTranscriber(api_key, excel_file=os.path.join(os.path.dirname(__file__), excel_file))

        # Audio buffer
        self.audio_buffer = []
        self.last_audio_ts: float = time.time()
        self.is_recording = False

    def _silence_detector(self, audio_data: np.ndarray) -> bool:
        """Detect if audio is silence"""
        if audio_data.size == 0:
            return True
        rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
        return rms < SILENCE_AMPLITUDE

    def _audio_callback(self, indata: np.ndarray, frames: int, _time, status) -> None:
        if status:
            print(f"Audio status: {status}")
            return

        now = time.time()

        # Add audio to buffer
        self.audio_buffer.extend(indata.flatten())

        # Check for silence
        if self._silence_detector(indata):
            if self.is_recording and (now - self.last_audio_ts) >= SILENCE_SECONDS:
                # Process the buffered audio
                self._process_audio_chunk()
                self.audio_buffer = []
                self.is_recording = False
        else:
            self.is_recording = True
            self.last_audio_ts = now

    def _process_audio_chunk(self) -> None:
        """Process a chunk of audio by transcribing it"""
        if not self.audio_buffer:
            return

        # Convert buffer to numpy array
        audio_data = np.array(self.audio_buffer, dtype=np.int16)

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_filename = temp_file.name

        try:
            # Write WAV file
            import wave
            with wave.open(temp_filename, 'wb') as wav_file:
                wav_file.setnchannels(CHANNELS)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_data.tobytes())

            # Transcribe using deep.py logic
            transcript_result = self.parser.transcribe_audio(temp_filename)

            # Extract transcript text
            transcript_text = ""
            if 'results' in transcript_result and 'channels' in transcript_result['results']:
                channel = transcript_result['results']['channels'][0]
                if 'alternatives' in channel:
                    transcript_text = channel['alternatives'][0].get('transcript', '')

            if transcript_text.strip():
                print(f"🎤 Live: {transcript_text}")

                # Parse for product code and quantity
                try:
                    pcode, remaining_text, confidence = self.parser.extract_pcode_and_remaining_text(transcript_text)
                    if pcode and confidence >= 1.0:  # Only show if exact match
                        # Parse the remaining text as quantity (convert words to numbers)
                        qty = self.parser.parse_quantity(remaining_text)
                        print(f"✅ Product: {pcode.upper()} | Qty: {qty or '-'}")
                    else:
                        print("❌ Can't recognize product code")
                except Exception as e:
                    print(f"❌ Parse error: {e}")
            else:
                print("🎤 Live: [silence or no speech detected]")

        except Exception as e:
            print(f"❌ Transcription error: {e}")
        finally:
            # Clean up temp file
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
                blocksize=int(SAMPLE_RATE * 0.1)  # 100ms blocks
            ):
                while not self.should_stop:
                    time.sleep(0.1)

        except KeyboardInterrupt:
            print("\n👋 Shutting down...")
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
            print("Error: Set DEEPGRAM_API_KEY env var or place API key in keys/deepgram.key")
            sys.exit(1)

    transcriber = LiveTranscriber(api_key)
    transcriber.run()


if __name__ == "__main__":
    main()


