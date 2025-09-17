# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os.path
from typing import List,  Tuple

import numpy as np

from utils.utils.utils import  read_yaml
from utils.utils.frontend import WavFrontend
from utils.utils.e2e_vad import E2EVadModel
import axengine as axe

class AX_Fsmn_vad:
    def __init__(self, model_dir, batch_size=1, max_end_sil=None):
            """Initialize VAD model for inference"""
            
            # Export model if needed
            model_file = os.path.join(model_dir, "vad.axmodel")
            
            # Load config and frontend
            config_file = os.path.join(model_dir, "vad/config.yaml")
            cmvn_file = os.path.join(model_dir, "vad/am.mvn")
            self.config = read_yaml(config_file)
            self.frontend = WavFrontend(cmvn_file=cmvn_file, **self.config["frontend_conf"])
            self.session = axe.InferenceSession(model_file, providers='AxEngineExecutionProvider')
            self.batch_size = batch_size
            self.vad_scorer = E2EVadModel(self.config["model_conf"])
            self.max_end_sil = max_end_sil if max_end_sil is not None else self.config["model_conf"]["max_end_silence_time"]

    def extract_feat(self, waveform_list):
        """Extract features from waveform"""
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        max_len = max(feats_len)
        padded_feats = [np.pad(f, ((0, max_len - f.shape[0]), (0, 0)), 'constant') for f in feats]
        feats = np.array(padded_feats).astype(np.float32)
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    def infer(self, feats: List) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference with ONNX Runtime"""
        # Get all input names from the model
        input_names = [input.name for input in self.session.get_inputs()]
        output_names = [x.name for x in self.session.get_outputs()]
        
        # Create input dictionary for all inputs
        input_dict = {}
        for i, (name, tensor) in enumerate(zip(input_names, feats)):
            input_dict[name] = tensor
            
        # Run inference with all inputs
        outputs = self.session.run(output_names, input_dict)
        scores, out_caches = outputs[0], outputs[1:]
        return scores, out_caches

    def __call__(self, wav_file, **kwargs):
        """Process audio file with sliding window approach"""
        # Load audio and prepare data
        # waveform = self.load_wav(wav_file)
        # waveform, _ = librosa.load(wav_file, sr=16000)
        waveform_list = [wav_file]
        waveform_nums = len(waveform_list)
        is_final = kwargs.get("kwargs", False)
        segments = [[]] * self.batch_size

        for beg_idx in range(0, waveform_nums, self.batch_size):
            vad_scorer = E2EVadModel(self.config["model_conf"])
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            waveform = waveform_list[beg_idx:end_idx]
            feats, feats_len = self.extract_feat(waveform)
            waveform = np.array(waveform)
            param_dict = kwargs.get("param_dict", dict())
            in_cache = param_dict.get("in_cache", list())
            in_cache = self.prepare_cache(in_cache)

            t_offset = 0
            step = int(min(feats_len.max(), 6000))
            for t_offset in range(0, int(feats_len), min(step, feats_len - t_offset)):
                if t_offset + step >= feats_len - 1:
                    step = feats_len - t_offset
                    is_final = True
                else:
                    is_final = False

                # Extract feature segment
                feats_package = feats[:, t_offset:int(t_offset + step), :]

                # Pad if it's the final segment
                if is_final:
                    pad_length = 6000 - int(step)
                    feats_package = np.pad(
                        feats_package,
                        ((0, 0), (0, pad_length), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )

                # Extract corresponding waveform segment
                waveform_package = waveform[
                    :,
                    t_offset * 160:min(waveform.shape[-1], (int(t_offset + step) - 1) * 160 + 400),
                ]

                # Pad waveform if it's the final segment
                if is_final:
                    expected_wave_length = 6000 * 160 + 240
                    current_wave_length = waveform_package.shape[-1]
                    pad_wave_length = expected_wave_length - current_wave_length
                    if pad_wave_length > 0:
                        waveform_package = np.pad(
                            waveform_package,
                            ((0, 0), (0, pad_wave_length)),
                            mode='constant',
                            constant_values=0
                        )

                # Run inference
                inputs = [feats_package]
                inputs.extend(in_cache)
                scores, out_caches = self.infer(inputs)
                in_cache = out_caches
                
                # Get VAD segments for this chunk
                segments_part = vad_scorer(
                    scores,
                    waveform_package,
                    is_final=is_final,
                    max_end_sil=self.max_end_sil,
                    online=False,
                )

                # Accumulate segments
                if segments_part:
                    for batch_num in range(0, self.batch_size):
                        segments[batch_num] += segments_part[batch_num]

        return segments

    def prepare_cache(self, in_cache: list = []):
        if len(in_cache) > 0:
            return in_cache
        fsmn_layers = 4
        proj_dim = 128
        lorder = 20
        for i in range(fsmn_layers):
            cache = np.zeros((1, proj_dim, lorder - 1, 1)).astype(np.float32)
            in_cache.append(cache)
        return in_cache

