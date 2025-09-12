import sys
import os
from pathlib import Path
from utils.vad_bin import Fsmn_vad
model_dir = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

wav_path = "wav/vad_example.wav"

model_save_dir = "output_dir"
datasets_dir = "datasets"
model = Fsmn_vad(model_dir, model_save_dir,datasets_dir)
result = model(wav_path)
print(result)
