import os
import sys
import asyncio
import json
import numpy as np
import sounddevice as sd
import websockets

# Import parser logic from your existing live_deep.py
from live_deep import ProductCodeTranscriber

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

# Deepgram API key
DEEPGRAM_API_KEY = (
    os.getenv("DEEPGRAM_API_KEY")
    or open(os.path.join("keys", "deepgram.key")).read().strip()
)


class LiveDeepStream:
    def __init__(self, api_key: str, excel_file: str = "productlist.xlsx"):
        self.api_key = api_key
        self.parser = ProductCodeTranscriber(api_key, excel_file)
        self.partial_text = ""

    async def _send_audio(self, ws):
        """Send continuous mic data to Deepgram"""
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE) as stream:
            print("🎤 Live Deep Fuzzy Transcriber — Speak product and quantity, pause to finalize.\n")
            while True:
                data, _ = stream.read(int(SAMPLE_RATE / 4))  # 250ms of audio
                await ws.send(data.tobytes())

    async def _receive_transcripts(self, ws):
        """Receive and display live transcripts"""
        async for message in ws:
            msg = json.loads(message)

            # Deepgram sends results in msg["channel"]["alternatives"]
            if "channel" not in msg or "alternatives" not in msg["channel"]:
                continue

            transcript = msg["channel"]["alternatives"][0].get("transcript", "")
            if not transcript:
                continue

            is_final = msg.get("is_final", False)

            if not is_final:
                # show real-time text on same line
                sys.stdout.write(f"\r🗣️ {transcript}")
                sys.stdout.flush()
            else:
                print(f"\n✅ Final: {transcript}")

                # Use your deep parser logic to extract code + qty
                try:
                    pcode, qty, confidence = self.parser.extract_pcode_and_qty(transcript)
                    conf_label = (
                        "EXACT"
                        if confidence == 1.0
                        else f"FUZZY({confidence:.2f})" if confidence > 0 else "NONE"
                    )

                    print(
                        f"🧩 Parsed -> ProductID: {pcode or 'Unknown'} | Qty: {qty or '-'} | Confidence: {conf_label}\n"
                    )
                except Exception as e:
                    print(f"❌ Parse error: {e}\n")

    async def run(self):
        """Run Deepgram websocket streaming"""
        uri = f"wss://api.deepgram.com/v1/listen?encoding=linear16&sample_rate={SAMPLE_RATE}"
        headers = {"Authorization": f"Token {self.api_key}"}

        async with websockets.connect(uri, extra_headers=headers) as ws:
            await asyncio.gather(
                self._send_audio(ws),
                self._receive_transcripts(ws),
            )


async def main():
    key_path = os.path.join("keys", "deepgram.key")
    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        try:
            with open(key_path, "r") as f:
                api_key = f.read().strip()
        except Exception:
            print("Error: Provide Deepgram key in keys/deepgram.key or DEEPGRAM_API_KEY env.")
            sys.exit(1)

    streamer = LiveDeepStream(api_key)
    try:
        await streamer.run()
    except KeyboardInterrupt:
        print("\n👋 Stopped by user.")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
