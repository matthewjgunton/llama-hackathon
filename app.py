import gradio as gr
import sounddevice as sd
import torch
from transformers import AutoProcessor, SeamlessM4Tv2Model
import warnings
from threading import Thread
from queue import Queue
import time
import traceback

warnings.filterwarnings("ignore")

class ContinuousTranslator:
    def __init__(self, target_language="spa", chunk_duration=5, sample_rate=16000):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.target_language = target_language
        self.is_running = False
        self.audio_queue = Queue()
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
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
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
        return torch.tensor(recording.T)

    def translate_audio(self, audio):
        """Translate audio chunk using the ML model"""
        try:
            audio_inputs = self.processor(audios=audio, return_tensors="pt")
            audio_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                          for k, v in audio_inputs.items()}

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
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def recording_worker(self):
        """Worker function to continuously record audio"""
        while self.is_running:
            audio = self.record_audio_chunk()
            self.audio_queue.put(audio)

    def start_translation(self):
        """Generator function that yields translations as they become available"""
        self.is_running = True
        self.record_thread = Thread(target=self.recording_worker)
        self.record_thread.start()
        
        full_transcript = ""
        
        try:
            while self.is_running:
                if not self.audio_queue.empty():
                    audio = self.audio_queue.get()
                    translation = self.translate_audio(audio)
                    if translation:
                        full_transcript += f"{translation} "
                        yield full_transcript
                time.sleep(0.1)
        finally:
            self.stop()

    def stop(self):
        """Stop the continuous recording and translation process"""
        self.is_running = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join()

translator_instance = None

def translate_audio_from_user(target_lang):
    global translator_instance
    translator_instance = ContinuousTranslator(target_language=target_lang)
    return gr.update(value="Starting translation...", visible=True), gr.update(visible=True)

def process_audio():
    global translator_instance
    if translator_instance:
        for transcript in translator_instance.start_translation():
            yield transcript

def stop_translation():
    global translator_instance
    if translator_instance:
        translator_instance.stop()
    return gr.update(value="Translation stopped.", visible=True), gr.update(visible=False)

with gr.Blocks() as app:
    gr.Markdown("## 🎤 Real-Time Speech Translator with Live Updates")
    
    with gr.Row():
        target_language = gr.Dropdown(
            label="Target Language",
            choices=["eng", "spa", "fra", "deu"],  # English, Spanish, French, German
            value="eng"
        )
    
    with gr.Row():
        start_button = gr.Button("Start Translation")
        stop_button = gr.Button("Stop Translation", visible=False)
    
    output_text = gr.TextArea(
        label="Translation Output",
        interactive=False,
        visible=True
    )

    start_button.click(
        fn=translate_audio_from_user,
        inputs=[target_language],
        outputs=[output_text, stop_button],
        queue=False
    ).success(
        fn=process_audio,
        outputs=output_text
    )
    
    stop_button.click(
        fn=stop_translation,
        outputs=[output_text, stop_button],
        queue=False
    )

if __name__ == "__main__":
    app.launch()