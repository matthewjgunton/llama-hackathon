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

model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
print(model.speech_encoder)