# 模型转换

## 创建虚拟环境
已验证环境：python3.10
```
python3.10 -m venv 3D-Speaker-MT
source 3D-Speaker-MT/bin/activate
```

## 安装依赖
```
pip install -r requiremets.txt
```

## 导出onnx模型并进行推理

### vad语音检测模型
```
cd vad
python export_vad_onnx.py
```
onnx模型及相关文件保存在output_dir,量化数据存在datasets
最终结果如下：
```
(3D-Speaker-MT) root@autodl-container-23a74daa49-c254949a:~/meeting_transc/model_convert/vad# python export_onnx.py 
Notice: ffmpeg is not installed. torchaudio is used to load audio
If you want to use ffmpeg backend to load audio, please install it by:
        sudo apt install ffmpeg # ubuntu
        # brew install ffmpeg # mac
Downloading Model from https://www.modelscope.cn to directory: /root/.cache/modelscope/hub/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch
2025-09-12 13:39:49,270 - modelscope - WARNING - Model revision not specified, use revision: v2.0.4
2025-09-12 13:39:49,348 - modelscope - INFO - Got 7 files, start to download ...
Downloading [config.yaml]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.19k/1.19k [00:00<00:00, 2.64kB/s]
Downloading [configuration.json]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 365/365 [00:00<00:00, 467B/s]
Downloading [am.mvn]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7.85k/7.85k [00:00<00:00, 10.2kB/s]
Downloading [README.md]: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 8.45k/8.45k [00:00<00:00, 10.6kB/s]
Downloading [fig/struct.png]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27.3k/27.3k [00:00<00:00, 33.3kB/s]
Downloading [example/vad_example.wav]: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.16M/2.16M [00:01<00:00, 2.25MB/s]
Downloading [model.pt]: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.64M/1.64M [00:01<00:00, 1.68MB/s]
Processing 7 items: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7.00/7.00 [00:01<00:00, 6.77it/s]
2025-09-12 13:39:50,383 - modelscope - INFO - Download model 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch' successfully.████████████████████████████████████████████████| 8.45k/8.45k [00:00<00:00, 10.7kB/s]
.onnx does not exist, begin to export onnx████████████████████████████████████████████████████████████████████████▎                                                      | 1.00M/1.64M [00:00<00:00, 1.08MB/s]
funasr version: 1.2.7.t.png]: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 27.3k/27.3k [00:00<00:00, 33.3kB/s]
WARNING:root:trust_remote_code: False
WARNING:root:trust_remote_code: False:  46%|█████████████████████████████████████████████████████████▉                                                                   | 1.00M/2.16M [00:00<00:01, 1.10MB/s]
results:
[[[70, 2340], [2620, 6200], [6480, 23670], [23950, 26250], [26780, 28990], [29950, 31430], [31750, 37600], [38210, 46900], [47310, 49630], [49910, 56460], [56740, 59540], [59820, 70550]]]
```

### 声纹识别模型campplus
```
cd campplus/
python export_campplus_onnx.py --model_id iic/speech_campplus_sv_zh_en_16k-common_advanced --experiment_path output_dir --target_onnx_file output_dir/model.onnx
```
onnx模型及相关文件保存在output_dir,量化数据存在datasets
当输入同一个人的两段说话音频时：
```
输出tensor:
tensor([0.6666])
```
当输入不同人之间的两段说话音频时：
```
输出tensor:
tensor([0.0572])
```
输出相似度越高表明两端音频属于同一个人的概率越大。

### 语音识别模型sensevoice
参考[sensevoice](https://github.com/ml-inory/sensevoice.axera/tree/main/model_convert)
seq_len是输入模型的特征长度，此工程demo目前设置为132。

## 导出axmodel

### vad 模型
```
cd vad
pulsar2 build --config vad_config.json
```
模型保存在vad_axmodel。

### campplus模型
```
cd campplus
pulsar2 build --config cam_config.json
```
模型保存在campplus_axmodel。

### sensevoice模型
同样参考[sensevoice](https://github.com/ml-inory/sensevoice.axera/tree/main/model_convert)




