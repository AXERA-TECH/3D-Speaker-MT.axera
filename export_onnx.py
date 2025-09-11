import sys
import os
from pathlib import Path
from utils.vad_bin import Fsmn_vad

#model_dir = "/home/hy/meeting_transcription/3D-Speaker-main/models/speech_fsmn_vad_zh-cn-16k-common-pytorch"
model_dir = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

wav_path = "vad_example.wav" # "{}/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/example/vad_example.wav".format(

wav_path_test = "wav/S_R004S03C01.wav"  # Example file, replace with your actual file path
model_save_dir = "output_dir"
model = Fsmn_vad(model_dir, model_save_dir)
result = model(wav_path_test)
print(result)
