#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os.path
from pathlib import Path
from typing import List, Union, Tuple
import torch
import numpy as np
import axengine as axe

try:
    import librosa
except ImportError:
    print("Warning: librosa not found. Please install it using 'pip install librosa'.")
    # Provide a fallback implementation if needed
    def load_wav_fallback(path, sr=None):
        import wave
        import numpy as np
        with wave.open(path, 'rb') as wf:
            num_frames = wf.getnframes()
            frames = wf.readframes(num_frames)
            return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0, wf.getframerate()

from utils.infer_utils import (
    CharTokenizer,
    get_logger,
    read_yaml,
)
from utils.frontend import WavFrontend
from utils.ctc_alignment import ctc_forced_align

logging = get_logger()


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class AX_SenseVoiceSmall:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        batch_size: int = 1,
        device_id: Union[str, int] = "-1",
        plot_timestamp_to: str = "",
        quantize: bool = False,
        intra_op_num_threads: int = 4,
        cache_dir: str = None,
        seq_len: int = 68,
        **kwargs,
    ):

        model_file = os.path.join(model_dir, "sensevoice.axmodel")
        config_file = os.path.join(model_dir, "sensevoice/config.yaml")
        cmvn_file = os.path.join(model_dir, "sensevoice/am.mvn")
        config = read_yaml(config_file)
        self.model_dir = model_dir
        # token_list = os.path.join(model_dir, "tokens.json")
        # with open(token_list, "r", encoding="utf-8") as f:
        #     token_list = json.load(f)

        # self.converter = TokenIDConverter(token_list)
        self.tokenizer = CharTokenizer()
        config["frontend_conf"]['cmvn_file'] = cmvn_file
        self.frontend = WavFrontend(**config["frontend_conf"])
        # self.ort_infer = OrtInferSession(
        #     model_file, device_id, intra_op_num_threads=intra_op_num_threads
        # )
        self.session = axe.InferenceSession(model_file, providers='AxEngineExecutionProvider')
        self.batch_size = batch_size
        self.blank_id = 0
        self.seq_len = seq_len

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}

    def __call__(self, 
                 wav_content: Union[str, np.ndarray, List[str]], 
                 language: str,
                 withitn: bool,
                 position_encoding: np.ndarray,
                 tokenizer=None,
                 **kwargs) -> List:
        """Enhanced model inference with additional features from model.py
        
        Args:
            wav_content: Audio data or path
            language: Language code for processing
            withitn: Whether to use ITN (inverse text normalization)
            position_encoding: Position encoding tensor
            tokenizer: Tokenizer for text conversion
            **kwargs: Additional arguments
        """
        # Start time tracking for metadata
        import time
        meta_data = {}
        time_start = time.perf_counter()
        
        # Load waveform data
        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)
        time_load = time.perf_counter()
        meta_data["load_data"] = f"{time_load - time_start:0.3f}"
        # Get key for result identification
        key = kwargs.get("key", ["wav_file"])
        if isinstance(wav_content, str):
            wav_name = os.path.splitext(os.path.basename(wav_content))[0]
            if key == ["wav_file"]:
                key = [wav_name]
        
        # Load queries from saved numpy files
        language_query = np.load(os.path.join(self.model_dir, f"{language}.npy"))
        textnorm_query = np.load(os.path.join(self.model_dir, "withitn.npy") if withitn 
                                 else os.path.join(self.model_dir, "woitn.npy"))
        event_emo_query = np.load(os.path.join(self.model_dir, "event_emo.npy"))

        # Concatenate queries to form input_query
        input_query = np.concatenate((language_query, event_emo_query, textnorm_query), axis=1)

        # Setup dataset directories for saving intermediate files
        dataset = "dataset"
        os.makedirs(dataset, exist_ok=True)
        speech_dir = os.path.join(dataset, "speech", language, "withitn" if withitn else "woitn")
        mask_dir = os.path.join(dataset, "masks", language, "withitn" if withitn else "woitn")
        pe_dir = os.path.join(dataset, "position_encoding", language, "withitn" if withitn else "woitn")
        os.makedirs(speech_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(pe_dir, exist_ok=True)
        
        # Process features
        results = []
        output_timestamp = kwargs.get("output_timestamp", False)
        ban_emo_unk = kwargs.get("ban_emo_unk", False)
        ibest_writer = None
        # 添加时间偏移变量，用于跟踪连续的时间戳
        time_offset = 0.0
        # 添加合并时间戳变量，用于存储同一音频文件的所有时间戳，仅保留必要功能
        merged_timestamps = []
        merged_words = []
        
        
        # Handle output_dir without using DatadirWriter (which is not available)
        output_dir = kwargs.get("output_dir")

        slice_len = self.seq_len - 4
        time_pre = time.perf_counter()
        meta_data["preprocess"] = f"{time_pre - time_load:0.3f}"
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])
            
            time_feat = time.perf_counter()
            meta_data["extract_feat"] = f"{time_feat - time_pre:0.3f}"

            for i in range(int(np.ceil(feats.shape[1] / slice_len))):
                sub_feats = np.concatenate([input_query, feats[:, i*slice_len : (i+1)*slice_len, :]], axis=1)
                feats_len[0] = sub_feats.shape[1]
                
                # 计算当前片段的实际长度（帧数）
                actual_slice_length = min(slice_len, feats.shape[1] - i*slice_len)

                if feats_len[0] < self.seq_len:
                    sub_feats = np.concatenate([sub_feats, np.zeros((1, self.seq_len - feats_len[0], 560), dtype=np.float32)], axis=1)

                masks = sequence_mask(torch.IntTensor([self.seq_len]), maxlen=self.seq_len, dtype=torch.float32)[:, None, :]
                masks = masks.numpy()
                
                # Run inference

                ctc_logits, encoder_out_lens = self.infer(sub_feats, masks, position_encoding)
                
                
                # Convert to torch tensor for processing
                ctc_logits = torch.from_numpy(ctc_logits).float()
                
                # Ban emotion unknown token if requested
                if ban_emo_unk:
                    ctc_logits[:, :, self.emo_dict["unk"]] = -float("inf")
                
                # Process results for each batch
                b, n, d = ctc_logits.size()
                if isinstance(key, (list, tuple)) and len(key) < b:
                    key = key * b
                    
                for j in range(b):
                    x = ctc_logits[j, : encoder_out_lens[j].item(), :]
                    yseq = x.argmax(dim=-1)
                    yseq = torch.unique_consecutive(yseq, dim=-1)

                    mask = yseq != self.blank_id
                    token_int = yseq[mask].tolist()

                    # Convert tokens to text
                    text = tokenizer.decode(token_int) if tokenizer is not None else str(token_int)
                    # 文本处理
                    # 简化文本处理，不执行合并操作
                    
                    # Write to output directory if provided
                    if ibest_writer is not None:
                        ibest_writer["text"][key[j]] = text
                    
                    if output_timestamp and tokenizer is not None:
                        # Process timestamps similar to model.py
                        from itertools import groupby
                        timestamp = []
                        tokens = tokenizer.text2tokens(text)[4:] if hasattr(tokenizer, 'text2tokens') else []
                        # If ctc_forced_align is available, calculate timestamps
                        if 'ctc_forced_align' in globals() or 'ctc_forced_align' in locals():
                            softmax = torch.nn.Softmax(dim=-1)
                            logits_speech = softmax(ctc_logits[j, 4:encoder_out_lens[j].item(), :])
                            
                            pred = logits_speech.argmax(-1).cpu()
                            logits_speech[pred == self.blank_id, self.blank_id] = 0
                            
                            try:
                    
                                # Convert numpy types to PyTorch tensors where needed
                                # Make sure encoder_out_lens is a torch.Tensor before calling .long()
                                tokens_tensor = torch.Tensor(token_int[4:]).unsqueeze(0).long()
                                
                                # Handle numpy int64 by converting to torch tensor first
                                if isinstance(encoder_out_lens[j], (np.integer, np.int64)):
                                    lens_tensor = torch.tensor(int(encoder_out_lens[j]-4))
                                else:
                                    lens_tensor = (encoder_out_lens[j]-4)
                                    if hasattr(lens_tensor, 'long'):
                                        lens_tensor = lens_tensor.long()
                                        
                                token_len = torch.tensor(len(token_int)-4).unsqueeze(0)
                                #token_len = torch.tensor(len(token_int)).unsqueeze(0)
                                
                                if hasattr(token_len, 'long'):
                                    token_len = token_len.long()
                                    
                                align = ctc_forced_align(
                                    logits_speech.unsqueeze(0).float(),
                                    tokens_tensor,
                                    lens_tensor,
                                    token_len,
                                    ignore_id=0
                                )
                                # 时间戳处理完成
                                # Process alignment
                                # Handle potential numpy type for slicing
                                if isinstance(encoder_out_lens[j], (np.integer, np.int64)):
                                    end_idx = int(encoder_out_lens[j]-4)
                                else:
                                    end_idx = encoder_out_lens[j]-4
                                    if hasattr(end_idx, 'item'):
                                        end_idx = end_idx.item()
                                
                                pred = groupby(align[0, :end_idx])
                                _start = 0
                                token_id = 0
                                # Convert ts_max to the right type for calculation
                                if isinstance(encoder_out_lens[j], (np.integer, np.int64)):
                                    ts_max = int(encoder_out_lens[j] - 4)
                                else:
                                    ts_max = encoder_out_lens[j] - 4
                                    if hasattr(ts_max, 'item'):
                                        ts_max = ts_max.item()
                                #timestamps_one_sentence = [] # Store timestamps for the current sentence
                                for pred_token, pred_frame in pred:
                                    _end = _start + len(list(pred_frame))
                                    if pred_token != 0 and token_id < len(tokens):
                                        # 计算时间戳，加上时间偏移以保持连续，精确保留两位小数
                                        ts_left = round(max((_start*60-30)/1000, 0) + time_offset, 2)
                                        ts_right = round(min((_end*60-30)/1000, (ts_max*60-30)/1000) + time_offset, 2)

                                        merged_timestamps.append([ts_left, ts_right])
                                        merged_words.append(tokens[token_id])
                                        #timestamps_one_sentence.append(ts_entry)
                                        token_id += 1
                                    _start = _end
                                #merged_timestamps.append(timestamps_one_sentence)
                                # 时间戳处理完成
                            except (ImportError, Exception) as e:
                                logging.warning(f"Timestamp calculation failed: {e}")
                        
                        result_i = {"key": key[j] if j < len(key) else f"result_{j}", "text": text, "timestamp": timestamp}
                    else:
                        result_i = {"key": key[j] if j < len(key) else f"result_{j}", "text": text}
                    
                    # 直接添加结果，重复处理将在export.py中进行
                    results.append(result_i)

                time_offset = round(time_offset + (actual_slice_length * 60 ) / 1000, 2)
        # 结果处理完成
        time_end = time.perf_counter()
        meta_data["total_time"] = f"{time_end - time_start:0.3f}"
        meta_data["inference_time"] = f"{time_end - time_feat:0.3f}"
        
        # 简化元数据处理，只保留时间戳
        if len(results) > 0 and output_timestamp and merged_timestamps:
            # 只保留时间戳数据，以便在export.py中处理
            meta_data["merged_timestamps"] = merged_timestamps
            meta_data["merged_words"] = merged_words
   
        return results, meta_data

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            try:
                # Use librosa if available
                if 'librosa' in globals():
                    waveform, _ = librosa.load(path, sr=fs)
                else:
                    # Use fallback implementation
                    waveform, native_sr = load_wav_fallback(path)
                    if fs is not None and native_sr != fs:
                        # Implement resampling if needed
                        print(f"Warning: Resampling from {native_sr} to {fs} is not implemented in fallback mode")
                return waveform
            except Exception as e:
                print(f"Error loading audio file {path}: {e}")
                # Return empty audio in case of error
                return np.zeros(1600, dtype=np.float32)

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)

            feat, feat_len = self.frontend.lfr_cmvn(speech)

            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(self, 
              feats: np.ndarray, 
              masks: np.ndarray,
              position_encoding: np.ndarray,
              ) -> Tuple[np.ndarray, np.ndarray]:
        #outputs = self.ort_infer([feats, masks, position_encoding])
        outputs =self.session.run(None, {
            'speech': feats,
            'masks': masks,
            'position_encoding': position_encoding
        })
        return outputs
