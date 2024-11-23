"""
Current issues:
- it sometimes misses information when the chunk is too small
    - testing: speak for 5 seconds straight and see what it sees
- difficulty with a web interface
- goal is for 2 people to talk to the website & it shows them what they said
    - is the first input enough to get the base language?
- 
- 
- 
"""
import sounddevice as sd
import soundfile as sf
import torch
import numpy as np
import time
from transformers import AutoProcessor, SeamlessM4Tv2Model
from threading import Thread
from queue import Queue
import warnings
warnings.filterwarnings("ignore")

class ContinuousTranslator:
    def __init__(self, target_language="spa", chunk_duration=5, sample_rate=16000):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.target_language = target_language
        self.is_running = False
        self.audio_queue = Queue()
        self.transcript = ""
        self.device = self._get_best_device()
        
        # Initialize ML models
        print("Loading ML models...")
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
        self.model.to(self.device)
        print("Models loaded!")

    def _get_best_device(self):
        """Determine the best available device for computation"""
        if torch.cuda.is_available():
            print("CUDA (GPU) is available")
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("Apple Silicon MPS is available")
            return torch.device("mps")
        else:
            print("Using CPU")
            return torch.device("cpu")

    def record_audio_chunk(self):
        """Record a chunk of audio"""
        recording = sd.rec(
            int(self.sample_rate * self.chunk_duration),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        return torch.tensor(recording.T)  # Send directly to correct device

    def translate_audio(self, audio):
        """Translate audio chunk using the ML model"""
        try:
            audio_inputs = self.processor(audios=audio, return_tensors="pt")
                # Move inputs to the correct device
            audio_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                              for k, v in audio_inputs.items()}
            
            ### testing if I can pull out the text and then continue

            output_tokens = self.model.generate(
                **audio_inputs,
                tgt_lang=self.target_language,
                generate_speech=False
            )
            translated_text = self.processor.decode(
                output_tokens[0].tolist()[0],
                skip_special_tokens=True
            )
            return translated_text
        except Exception as e:
            return f"Translation error: {str(e)}"
        finally:
            # Clear CUDA cache if using GPU
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def recording_worker(self):
        """Worker function to continuously record audio"""
        while self.is_running:
            audio = self.record_audio_chunk()
            self.audio_queue.put(audio)

    def translation_worker(self):
        """Worker function to process and translate audio chunks"""
        while self.is_running:
            if not self.audio_queue.empty():
                audio = self.audio_queue.get()
                translation = self.translate_audio(audio)
                print(f"\nTranslation: {translation}")
                self.transcript += translation
            time.sleep(0.1)  # Small delay to prevent CPU overuse

    def start(self):
        """Start the continuous recording and translation process"""
        self.is_running = True
        
        # Create and start worker threads
        self.record_thread = Thread(target=self.recording_worker)
        self.translate_thread = Thread(target=self.translation_worker)
        
        self.record_thread.start()
        self.translate_thread.start()
        
        print(f"Started continuous translation to {self.target_language}")
        print("Press Ctrl+C to stop...")

    def stop(self):
        """Stop the continuous recording and translation process"""
        self.is_running = False
        self.record_thread.join()
        self.translate_thread.join()
        print("\nStopped continuous translation")

# Example usage

# inference llama3.2 and see if this is likely a scam

def main():
    translator = ContinuousTranslator(
        target_language="spa",  # Spanish
        chunk_duration=2,       # 2 seconds
        sample_rate=16000      # 16kHz
    )

    try:
        translator.start()
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        translator.stop()

if __name__ == "__main__":
    main()