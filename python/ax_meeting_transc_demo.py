#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os
import torch
import argparse
from model import  SinusoidalPositionEncoder
from utils.ax_model_bin import AX_SenseVoiceSmall
from utils.ax_vad_bin import AX_Fsmn_vad 
from utils.vad_utils import merge_vad
from utils.ax_cam_bin import AX_SpeakerEmbeddingInference, do_clustering, distribute_spk, get_trans_sentence_sensevoice
from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
from utils.ax_cam_bin import chunk
import time
import librosa
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(description="SenseVoice inference script")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=132, help="Sequence length for model") #68 ,132
    parser.add_argument("--wav_file", type=str, default="wav/vad_example.wav",help="Input wav file")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seq_len = args.seq_len
    model_path = args.output_dir
    os.makedirs(model_path, exist_ok=True)

    print(f"Initializing  model...")

    ax_model_dir = "ax_model"
    total_inference_start = time.time()
   
    model_vad = AX_Fsmn_vad(ax_model_dir)
    speaker_model = AX_SpeakerEmbeddingInference(model_dir=ax_model_dir)
    embed = SinusoidalPositionEncoder()
    position_encoding = embed.get_position_encoding(torch.randn(1, seq_len, 560)).numpy()
    model_bin = AX_SenseVoiceSmall(ax_model_dir, seq_len=seq_len)

    # build tokenizer
    print(f"Loading tokenizer...")
    tokenizer = None
    tokenizer_path = os.path.join(ax_model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
    tokenizer = SentencepiecesTokenizer(bpemodel=tokenizer_path)

    # Set up audio file for processing
    wav_file = args.wav_file #S_R004S03C01

    print(f"Running inference on example file...")
    
    withitn = True
    norm_type = "withitn" if withitn else "woitn"
    print(f"\nProcessing with text normalization: {norm_type}")
        
    language = "auto"
    print(f"\n--- Processing language: {language} ---")
    print(f"Processing file: {wav_file}")
    inference_start = time.time()
    # 加载音频数据
    
    speech, fs = librosa.load(wav_file, sr=None)
    # 检查采样率，如果不是16kHz则进行重采样
    if fs != 16000:
        print(f"Resampling audio from {fs}Hz to 16000Hz")
        speech = librosa.resample(y=speech, orig_sr=fs, target_sr=16000)
        fs = 16000
    audio_duration = librosa.get_duration(y=speech, sr=fs)
    speech_lengths = len(speech)

    try:
        #增加vad model 推理及处理
        vad_start_time = time.time()
        res_vad = model_vad(speech)[0]
        vad_segments = merge_vad(res_vad, 15 * 1000)  #短语音段合并 # vad_segments: [[0, 6480], [6480, 23670], [23670, 38210], [38210, 49910], [49910, 59820], [59820, 70550]]
        vad_time_cost = time.time() - vad_start_time
        print(f"VAD processing time: {vad_time_cost:.2f} seconds")
        
        # emb_extraction
        vad_time = [[vad_t[0]/1000, vad_t[1]/1000] for vad_t in res_vad]
        chunks = [c for (st, ed) in vad_time for c in chunk(st, ed)]

        # Extract speaker embeddings for each chunk
        print("Extracting speaker embeddings...")
        speaker_start_time = time.time()
        # import pdb 
        # pdb.set_trace()
        embeddings = speaker_model(speech, fs, chunks=chunks)
        speaker_time_cost = time.time() - speaker_start_time
        print(f"Speaker embedding extraction time: {speaker_time_cost:.2f} seconds")
        print(f"Generated embeddings shape: {embeddings.shape}")

        clustering_start_time = time.time()
        speaker_num, diar_results = do_clustering(chunks, embeddings, speaker_num=None)
        clustering_time_cost = time.time() - clustering_start_time
        print(f"Speaker clustering time: {clustering_time_cost:.2f} seconds")
        
        # print(f"VAD segments detected: {len(vad_segments)}")
        
        # 存储所有分片结果
        all_results = []
        all_metadata = {}
        
        # 遍历每个VAD片段并处理
        asr_start_time = time.time()
        for i, segment in enumerate(vad_segments):
            segment_start, segment_end = segment
            # 从原始音频中提取该片段
            start_sample = int(segment_start / 1000 * fs)
            end_sample = min(int(segment_end / 1000 * fs), speech_lengths)
            segment_speech = speech[start_sample:end_sample]
            
            # 计算时间偏移量（毫秒转秒）
            time_offset_sec = segment_start / 1000.0
            
            # 为当前片段创建临时文件
            segment_filename = f"temp_segment_{i}.wav"
            sf.write(segment_filename, segment_speech, fs)
            
            # 对当前片段进行识别
            try:
                segment_res, segment_meta = model_bin(
                    segment_filename, 
                    language, 
                    withitn, 
                    position_encoding, 
                    tokenizer=tokenizer,
                    output_timestamp=True,
                    ban_emo_unk=False,
                    output_dir=model_path,
                    key=[f"{os.path.basename(wav_file)}_segment_{i}"]
                )

                if "merged_words" in segment_meta:
                    if "merged_words" not in all_metadata:
                        all_metadata["merged_words"] = []
                    all_metadata["merged_words"].extend(segment_meta["merged_words"])
                    
                if "merged_timestamps" in segment_meta:
                    if "merged_timestamps" not in all_metadata:
                        all_metadata["merged_timestamps"] = []
                    adjusted_timestamps = [[min(ts[0] + time_offset_sec, audio_duration), min(ts[1] + time_offset_sec, audio_duration)]  
                                        for ts in segment_meta["merged_timestamps"]]  # 确保不超过音频总时长
                    all_metadata["merged_timestamps"].extend(adjusted_timestamps)
                
                if os.path.exists(segment_filename):
                    os.remove(segment_filename)
                    
            except Exception as e:
                if os.path.exists(segment_filename):
                    os.remove(segment_filename)
                raise
    
        output_asr = {
            "merged_words": all_metadata.get("merged_words", []),
            "merged_timestamps": all_metadata.get("merged_timestamps", [])
        }

    except Exception as e:
        raise

    asr_time_cost = time.time() - asr_start_time
    print(f"ASR processing time: {asr_time_cost:.2f} seconds")

    asr_timestamps = get_trans_sentence_sensevoice(output_asr)
    sentence_info_with_spk = distribute_spk(asr_timestamps, diar_results)
    inference_time_cost = time.time() - inference_start
    inference_time_cost_all = time.time() - total_inference_start
    rtf = inference_time_cost / audio_duration
    print(f"Inference time for {wav_file}: {inference_time_cost:.2f} seconds")
    print(f"load model  + Inference time for {wav_file}: {inference_time_cost_all:.2f} seconds")
    print(f"Audio duration: {audio_duration:.2f} seconds")
    print(f"RTF: {rtf:.2f}")

    # Save Results
    output_trans_path = os.path.join(args.output_dir, f"{wav_file.split('/')[-1]}.txt")
    with open(output_trans_path, 'w', encoding='utf-8') as f:
        for text_string, timeinterval, spk in sentence_info_with_spk:
            f.write(f'Speaker_{spk}: [{timeinterval[0]:.3f} {timeinterval[1]:.3f}] {text_string}\n')
