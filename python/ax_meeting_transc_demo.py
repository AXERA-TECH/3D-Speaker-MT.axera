#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os
import torch
import argparse

# Add parent directory to path to find local imports
#sys.path.append(str(Path(__file__).parent.parent))
from model import  SinusoidalPositionEncoder
from utils.ax_model_bin import AX_SenseVoiceSmall
from utils.ax_vad_bin import AX_Fsmn_vad 
from utils.vad_utils import merge_vad
from utils.ax_cam_bin import AX_SpeakerEmbeddingInference
from utils.ax_cam_bin import do_clustering
import time


def parse_args():
    parser = argparse.ArgumentParser(description="SenseVoice inference script")
    parser.add_argument("--model_dir", type=str, default="iic/SenseVoiceSmall", help="Path to the model directory")
    parser.add_argument("--output_dir", type=str, default="./output_dir", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=132, help="Sequence length for model") #68 ,132
    parser.add_argument("--output_timestamp", action="store_true", help="Output timestamps for each word")
    parser.add_argument("--ban_emo_unk", action="store_true", help="Ban unknown emotion token")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    seq_len = args.seq_len
    model_path = args.output_dir
	os.makedirs(model_path, exist_ok=True)
    print(f"Initializing SenseVoiceSmallONNX model...")

    ax_model_dir = "ax_model"

    embed = SinusoidalPositionEncoder()
    position_encoding = embed.get_position_encoding(torch.randn(1, seq_len, 560)).numpy()

    model_bin = AX_SenseVoiceSmall(ax_model_dir, seq_len=seq_len, quantize=False)

    model_dir = ax_model_dir
    model_vad = AX_Fsmn_vad(model_dir)

    model_cam_dir = ax_model_dir

    # build tokenizer
    print(f"Loading tokenizer...")
    tokenizer = None
    tokenizer_path = os.path.join(ax_model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")

    from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
    tokenizer = SentencepiecesTokenizer(bpemodel=tokenizer_path)


    # Set up audio file for processing
    vad_example = ["wav/vad_example.wav"] #S_R004S03C01
    data_dir = {
        "auto": vad_example
    }

    # Run inference on example file
    print(f"Running inference on example file...")
    total_inference_start = time.time()
    
    # for withitn in [True, False]:
    withitn = True
    norm_type = "withitn" if withitn else "woitn"
    print(f"\nProcessing with text normalization: {norm_type}")
        
    language = "auto"
    print(f"\n--- Processing language: {language} ---")
    for wav_file in data_dir[language]:
        if not os.path.exists(wav_file):
            print(f"Skipping non-existent file: {wav_file}")
            continue
            
        print(f"Processing file: {wav_file}")
        inference_start = time.time()

        try:
            #增加vad model 推理及处理
            res_vad = model_vad(wav_file)[0]
            vad_segments = merge_vad(res_vad, 15 * 1000)  #短语音段合并 # vad_segments: [[0, 6480], [6480, 23670], [23670, 38210], [38210, 49910], [49910, 59820], [59820, 70550]]
            
            # emb_extraction
            from utils.ax_cam_bin import chunk
            vad_time = [[vad_t[0]/1000, vad_t[1]/1000] for vad_t in res_vad]
            chunks = [c for (st, ed) in vad_time for c in chunk(st, ed)]

            # Initialize speaker embedding model
            speaker_model = AX_SpeakerEmbeddingInference(
                model_dir=model_cam_dir,  # 替换为实际的模型路径
            )
            
            # Extract speaker embeddings for each chunk
            print("Extracting speaker embeddings...")
            embeddings = speaker_model(wav_file, chunks=chunks)
            print(f"Generated embeddings shape: {embeddings.shape}")

            speaker_num, diar_results = do_clustering(chunks, embeddings, speaker_num=None)
            
            
            print(f"VAD segments detected: {len(vad_segments)}")
            
            # 加载音频数据
            try:
                import librosa
                speech, fs = librosa.load(wav_file, sr=None)
                audio_duration = librosa.get_duration(y=speech, sr=fs)
            except ImportError:
                from utils.ax_model_bin import load_wav_fallback
                speech, fs = load_wav_fallback(wav_file)
                
            speech_lengths = len(speech)
            
            # 存储所有分片结果
            all_results = []
            all_metadata = {}
            
            # 遍历每个VAD片段并处理
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
                import soundfile as sf
                sf.write(segment_filename, segment_speech, fs)
                
                # 对当前片段进行识别
                try:
                    segment_res, segment_meta = model_bin(
                        segment_filename, 
                        language, 
                        withitn, 
                        position_encoding, 
                        tokenizer=tokenizer,
                        output_timestamp=args.output_timestamp,
                        ban_emo_unk=args.ban_emo_unk,
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
                        adjusted_timestamps = [[ts[0] + time_offset_sec, min(ts[1] + time_offset_sec, audio_duration)] 
                                            for ts in segment_meta["merged_timestamps"]]
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

        from utils.ax_cam_bin import distribute_spk, get_trans_sentence_sensevoice

        asr_timestamps = get_trans_sentence_sensevoice(output_asr)
        sentence_info_with_spk = distribute_spk(asr_timestamps, diar_results)

        # Save Results
        output_trans_path = os.path.join(args.output_dir, f"{wav_file.split('/')[-1]}.txt")
        with open(output_trans_path, 'w', encoding='utf-8') as f:
            for text_string, timeinterval, spk in sentence_info_with_spk:
                f.write(f'Speaker_{spk}: [{timeinterval[0]:.3f} {timeinterval[1]:.3f}] {text_string}\n')
