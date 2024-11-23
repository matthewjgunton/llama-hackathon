import sounddevice as sd
import numpy as np
import requests
import threading
import time
import base64
import io
import soundfile as sf
import json
from datetime import datetime
import colorama
from colorama import Fore, Style

class RealtimeTranslationClient:
    def __init__(self, server_url, chunk_duration=5, sample_rate=16000, target_language="spa"):
        """
        Initialize the client with recording parameters and server connection
        """
        self.server_url = server_url.rstrip('/')
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.target_language = target_language
        self.is_running = False
        
        # Initialize colorama for colored output
        colorama.init()
        
        # Test server connection
        self.test_server_connection()
        
    def test_server_connection(self):
        """Test if the translation server is accessible"""
        try:
            response = requests.get(f"{self.server_url}/health")
            if response.status_code == 200:
                print(f"{Fore.GREEN}✓ Connected to translation server{Style.RESET_ALL}")
            else:
                raise Exception("Server returned non-200 status code")
        except Exception as e:
            print(f"{Fore.RED}✗ Failed to connect to translation server: {e}{Style.RESET_ALL}")
            raise

    def record_audio_chunk(self):
        """Record a chunk of audio and convert it to the right format"""
        print(f"{Fore.YELLOW}Recording...{Style.RESET_ALL}", end='\r')
        
        # Record audio
        recording = sd.rec(
            int(self.sample_rate * self.chunk_duration),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        
        # Convert to bytes
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, recording, self.sample_rate, format='WAV')
        audio_bytes = audio_buffer.getvalue()
        
        return base64.b64encode(audio_bytes).decode()

    def send_for_translation(self, audio_base64):
        """Send audio chunk to server for translation"""
        try:
            response = requests.post(
                f"{self.server_url}/translate",
                json={
                    'audio_data': audio_base64,
                    'target_language': self.target_language
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('translation', '')
            else:
                print(f"{Fore.RED}Error: Server returned status {response.status_code}{Style.RESET_ALL}")
                return None
                
        except Exception as e:
            print(f"{Fore.RED}Error sending audio for translation: {e}{Style.RESET_ALL}")
            return None

    def recording_worker(self):
        """Worker function to continuously record and translate audio"""
        while self.is_running:
            # Record audio chunk
            audio_base64 = self.record_audio_chunk()
            
            # Send for translation
            translation = self.send_for_translation(audio_base64)
            
            # Display translation if received
            if translation:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\n{Fore.CYAN}[{timestamp}] Translation:{Style.RESET_ALL} {translation}")

    def start(self):
        """Start the continuous recording and translation process"""
        self.is_running = True
        
        print(f"\n{Fore.GREEN}Starting real-time translation to {self.target_language}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Press Ctrl+C to stop...{Style.RESET_ALL}\n")
        
        # Create and start worker thread
        self.record_thread = threading.Thread(target=self.recording_worker)
        self.record_thread.start()

    def stop(self):
        """Stop the continuous recording and translation process"""
        self.is_running = False
        if hasattr(self, 'record_thread'):
            self.record_thread.join()
        print(f"\n{Fore.GREEN}Stopped translation client{Style.RESET_ALL}")

def main():
    # Configuration
    SERVER_URL = "http://localhost:5000"  # Change this to your server URL
    CHUNK_DURATION = 5  # seconds
    SAMPLE_RATE = 16000
    TARGET_LANGUAGE = "spa"  # Change this to your desired target language
    
    # Create and start client
    client = RealtimeTranslationClient(
        server_url=SERVER_URL,
        chunk_duration=CHUNK_DURATION,
        sample_rate=SAMPLE_RATE,
        target_language=TARGET_LANGUAGE
    )
    
    try:
        client.start()
        
        # Keep the main thread running
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\nStopping translation client...")
        client.stop()
    
    except Exception as e:
        print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
        client.stop()

if __name__ == "__main__":
    main()