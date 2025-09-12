import os
import sys
import numpy as np
import torch
import torchaudio

sys.path.append('%s'%os.path.dirname(__file__))

from speakerlab.utils.builder import build
from speakerlab.utils.config import Config
from speakerlab.utils.utils import circle_pad
from speakerlab.process.processor import FBank

import axengine as axe

def get_trans_sentence_sensevoice(output_asr):
    """Get transcription with timestamps from ASR"""
    sentence_info = [[]]
    punc_pattern = r'[,.!?;:"\-—…、，。！？；：""'']'
    
    words = output_asr['merged_words']
    #text = asr_result[0]['text']
    timestamp = output_asr['merged_timestamps']
    assert len(timestamp) == len(words)
    text_pt = 0
    
    # 遍历每个单词及其时间戳
    for i, wd in enumerate(words):
        # 如果当前单词是标点符号，将其与前一个单词合并
        if wd in punc_pattern and sentence_info and sentence_info[-1]:
            # 合并标点符号到前一个单词
            prev_word, prev_ts = sentence_info[-1][-1]
            sentence_info[-1][-1] = [prev_word + wd, [prev_ts[0], timestamp[i][1]]]
            
            # # 如果标点是句子结束标点，开始新句子
            if i < len(words) - 1:
                sentence_info.append([])
        else:
            # 对于非标点单词，直接添加到当前句子
            sentence_info[-1].append([wd, timestamp[i]])
    return sentence_info


def match_spk(sentence, output_field_labels):
    """Match speaker ID with transcription segments"""
    if len(sentence) == 0:
        return []
        
    st_sent = sentence[0][1][0]
    ed_sent = sentence[-1][1][1]
    overlap_per_spk = {}
    
    for st_spk, ed_spk, spk in output_field_labels:
        overlap_dur = min(ed_sent, ed_spk) - max(st_sent, st_spk)
        if spk not in overlap_per_spk:
            overlap_per_spk[spk] = 0
        if overlap_dur > 0:
            overlap_per_spk[spk] += overlap_dur
            
    overlap_per_spk_list = [[spk, overlap_per_spk[spk]] for spk in overlap_per_spk if overlap_per_spk[spk] > 0]
    overlap_per_spk_list = sorted(overlap_per_spk_list, key=lambda x:x[1], reverse=True)
    overlap_per_spk_list = [i[0] for i in overlap_per_spk_list]
    
    return overlap_per_spk_list
def distribute_spk(sentence_info, output_field_labels):
    """Distribute speaker IDs to transcription"""
    last_spk = 0
    for sentence in sentence_info:
        main_spks = match_spk(sentence, output_field_labels)
        main_spk = main_spks[0] if len(main_spks) > 0 else last_spk
        
        for i, wd in enumerate(sentence):
            wd_spks = match_spk([wd], output_field_labels)
            if main_spk in wd_spks:
                sentence[i].append(main_spk)
            elif len(wd_spks) > 0:
                sentence[i].append(wd_spks[0])
            else:
                sentence[i].append(last_spk)
        last_spk = sentence[-1][2]
        
    if len(sentence_info) == 0:
        return []
        
    # Merge consecutive segments from same speaker
    sentence_info = [j for i in sentence_info for j in i]
    sentence_info_with_spk_merge = [sentence_info[0]]
    
    for i in sentence_info[1:]:
        if (i[2] == sentence_info_with_spk_merge[-1][2] and 
            i[1][0] < sentence_info_with_spk_merge[-1][1][1] + 2):
            sentence_info_with_spk_merge[-1][0] += i[0]
            sentence_info_with_spk_merge[-1][1][1] = i[1][1]
        else:
            sentence_info_with_spk_merge.append(i)
            
    return sentence_info_with_spk_merge
def get_cluster_backend():
    conf = {
        'cluster':{
            'obj': 'speakerlab.process.cluster.CommonClustering',
            'args':{
                'cluster_type': 'spectral',
                'mer_cos': 0.8,
                'min_num_spks': 1,
                'max_num_spks': 15,
                'min_cluster_size': 4,
                'oracle_num': None,
                'pval': 0.012,
            }
        }
    }
    config = Config(conf)
    return build('cluster', config)


def chunk(st, ed, dur=1.5, step=0.75):
        chunks = []
        subseg_st = st
        while subseg_st + dur < ed + step:
            subseg_ed = min(subseg_st + dur, ed)
            chunks.append([subseg_st, subseg_ed])
            subseg_st += step
        return chunks

def compressed_seg(seg_list):
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed, cluster_id = seg
        if i == 0:
            new_seg_list.append([seg_st, seg_ed, cluster_id])
        elif cluster_id == new_seg_list[-1][2]:
            if seg_st > new_seg_list[-1][1]:
                new_seg_list.append([seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][1] = seg_ed
        else:
            if seg_st < new_seg_list[-1][1]:
                p = (new_seg_list[-1][1]+seg_st) / 2
                new_seg_list[-1][1] = p
                seg_st = p
            new_seg_list.append([seg_st, seg_ed, cluster_id])
    return new_seg_list

def do_clustering(chunks, embeddings, speaker_num=None):
        cluster = get_cluster_backend()
        cluster_labels = cluster(
            embeddings, 
            speaker_num = speaker_num if speaker_num is not None else speaker_num
        )
        speaker_num = cluster_labels.max()+1
        output_field_labels = [[i[0], i[1], int(j)] for i, j in zip(chunks, cluster_labels)]
        output_field_labels = compressed_seg(output_field_labels)
        return speaker_num, output_field_labels



class AX_SpeakerEmbeddingInference:
    def __init__(self, model_dir,  intra_op_num_threads=4):
        """Initialize speaker embedding model for inference"""           
        model_file = os.path.join(model_dir, "campplus.axmodel")
        self.session = axe.InferenceSession(model_file, providers='AxEngineExecutionProvider')

    def infer(self, feats: np.ndarray) -> np.ndarray:
        """Run inference with ONNX Runtime"""
        # Run inference
        outputs = self.session.run(None, {'feature': feats})
        return outputs[0]

    def __call__(self, wav_file, chunks=None, **kwargs):
        """Process audio file with chunks
        Args:
            wav_file: path to wav file
            chunks: list of [start_time, end_time] in seconds
        """
        # Load wav file
        wav, fs = torchaudio.load(wav_file)
        if wav.shape[0] > 1:
            wav = wav[0:1]  # Convert to mono if stereo

        wavs = [wav[0, int(st * fs):int(ed * fs)] for st, ed in chunks]
        # Pad all chunks to same length
        max_len = max([x.shape[0] for x in wavs])
        max_len = max(max_len, 57900)  # feats_batch[1,360,80] --> wavs[1, 57900]    better than max_len=24001
        #wavs = [torch.nn.functional.pad(x, (0, max_len - x.shape[1])) for x in wavs]
        wavs = [circle_pad(x, max_len) for x in wavs]
        wavs = torch.stack(wavs).unsqueeze(1)
        #wavs = torch.cat(wavs, dim=0)  # [num_chunks, samples]
        
        # Process in batches
        batch_size = 1 # onnx推理batchsize=1
        embeddings = []

        for i in range(0, len(wavs), batch_size):
            batch_wavs = wavs[i:i+batch_size]
            
            # Process entire batch at once
            feature_extractor = FBank(80, fs, mean_nor=True)
            feats_batch = torch.vmap(feature_extractor)(batch_wavs)
            
            # Adjust feature shape for all samples in batch
            if feats_batch.shape[1] >= 360:  # Use fixed 360 frames
                feats_batch = feats_batch.narrow(1, 0, 360)
            else:
                target_shape = list(feats_batch.shape)
                target_shape[1] = 360
                feats_batch = feats_batch.new_full(target_shape, 0.0)
            
            # Convert to numpy and run inference for batch
            feats_batch = feats_batch.numpy()
            embeddings_batch = self.infer(feats_batch)
            embeddings.append(embeddings_batch)
        
        # Concatenate all embeddings
        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings
