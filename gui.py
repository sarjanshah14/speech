import os
import sys
import threading
import tkinter as tk
from tkinter import scrolledtext
import builtins

from live_deep import LiveTranscriber  # make sure this matches your filename


class GuiApp:
    def __init__(self, master):
        self.master = master
        master.title("🎤 Live Product Code Transcriber")
        master.geometry("700x500")
        master.configure(bg="#1e1e1e")

        # Title
        tk.Label(
            master,
            text="Live Product Code Transcriber",
            font=("Segoe UI", 16, "bold"),
            bg="#1e1e1e",
            fg="white"
        ).pack(pady=10)

        # Text box for logs/transcription
        self.text_box = scrolledtext.ScrolledText(
            master,
            wrap=tk.WORD,
            width=80,
            height=20,
            bg="#252526",
            fg="white",
            insertbackground="white"
        )
        self.text_box.pack(padx=10, pady=10, fill="both", expand=True)

        # Buttons frame
        frame = tk.Frame(master, bg="#1e1e1e")
        frame.pack(pady=10)

        self.start_button = tk.Button(
            frame,
            text="Start Listening",
            command=self.start_listening,
            bg="#007acc",
            fg="white",
            width=15
        )
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = tk.Button(
            frame,
            text="Stop",
            command=self.stop_listening,
            bg="#d32f2f",
            fg="white",
            width=15,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=10)

        self.transcriber = None
        self.transcriber_thread = None

    def log(self, message: str):
        """Append a message to the GUI text box"""
        self.text_box.insert(tk.END, message + "\n")
        self.text_box.see(tk.END)

    def start_listening(self):
        """Start background transcription"""
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.text_box.delete(1.0, tk.END)
        self.log("🎧 Starting live transcription...")

        # Load Deepgram key
        api_key_path = os.path.join(os.path.dirname(__file__), "keys", "deepgram.key")
        try:
            with open(api_key_path, "r") as f:
                api_key = f.read().strip()
        except Exception:
            self.log("❌ Could not load API key. Make sure keys/deepgram.key exists.")
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            return

        self.transcriber = LiveTranscriber(api_key)

        # Redirect print to GUI
        def gui_print(*args, **kwargs):
            text = " ".join(map(str, args))
            self.master.after(0, lambda: self.log(text))

        sys.stdout.write = lambda text: self.master.after(0, lambda: self.log(text.strip()))
        sys.stderr.write = sys.stdout.write

        # Replace built-in print safely
        self._original_print = builtins.print
        builtins.print = gui_print

        def run_transcriber():
            try:
                self.transcriber.run()
            except Exception as e:
                self.master.after(0, lambda: self.log(f"❌ Error: {e}"))
            finally:
                # Restore original print
                builtins.print = self._original_print
                self.master.after(0, lambda: self.stop_listening())

        # Run in background
        self.transcriber_thread = threading.Thread(target=run_transcriber, daemon=True)
        self.transcriber_thread.start()

    def stop_listening(self):
        """Stop transcription"""
        if self.transcriber:
            self.transcriber.should_stop = True
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.log("🛑 Stopped listening.")


if __name__ == "__main__":
    root = tk.Tk()
    app = GuiApp(root)
    root.mainloop()
