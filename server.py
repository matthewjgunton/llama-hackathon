from flask import Flask, request, jsonify, Response
import sounddevice as sd
import soundfile as sf
import torch
import numpy as np
import time
from transformers import AutoProcessor, SeamlessM4Tv2Model
from threading import Thread
from queue import Queue
import warnings
import base64
import io
warnings.filterwarnings("ignore")

app = Flask(__name__)

class TranslationService:
    def __init__(self):
        print("Initializing Translation Service...")
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
        self.model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
        print("Models loaded!")
        
    def translate_audio_chunk(self, audio_data, target_language="spa"):
        """Translate a single audio chunk"""
        try:
            # Convert audio data to tensor
            audio_tensor = torch.tensor(audio_data)
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
                
            # Process and translate
            audio_inputs = self.processor(audios=audio_tensor, return_tensors="pt")
            output_tokens = self.model.generate(
                **audio_inputs,
                tgt_lang=target_language,
                generate_speech=False
            )
            translated_text = self.processor.decode(
                output_tokens[0].tolist()[0],
                skip_special_tokens=True
            )
            return translated_text
        except Exception as e:
            return f"Translation error: {str(e)}"

# Initialize the translation service
translation_service = TranslationService()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"})

@app.route('/translate', methods=['POST'])
def translate():
    """
    Endpoint to translate audio
    Expects JSON with:
    - audio_data: base64 encoded audio data
    - target_language: target language code (default: spa)
    """
    try:
        data = request.get_json()
        
        if not data or 'audio_data' not in data:
            return jsonify({"error": "No audio data provided"}), 400
            
        # Get target language, default to Spanish
        target_language = data.get('target_language', 'spa')
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(data['audio_data'])
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
        
        # Perform translation
        translation = translation_service.translate_audio_chunk(
            audio_data,
            target_language=target_language
        )
        
        return jsonify({
            "translation": translation,
            "target_language": target_language
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/batch_translate', methods=['POST'])
def batch_translate():
    """
    Endpoint for batch translation of multiple audio chunks
    Expects JSON with:
    - audio_chunks: list of base64 encoded audio data
    - target_language: target language code (default: spa)
    """
    try:
        data = request.get_json()
        
        if not data or 'audio_chunks' not in data:
            return jsonify({"error": "No audio chunks provided"}), 400
            
        target_language = data.get('target_language', 'spa')
        translations = []
        
        for chunk in data['audio_chunks']:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(chunk)
            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
            
            # Translate chunk
            translation = translation_service.translate_audio_chunk(
                audio_data,
                target_language=target_language
            )
            translations.append(translation)
        
        return jsonify({
            "translations": translations,
            "target_language": target_language
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)