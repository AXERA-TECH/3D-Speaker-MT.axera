---
license: mit
language:
- en
- zh
pipeline_tag: audio-text-to-text
base_model:
- FunAudioLLM/SenseVoiceSmall
tags:
- VAD
- ASR
---


# 3D-Speaker-MT.axera

meeting transcription demo on Axera

- [x] Python 示例
- [ ] C++ 示例

## Convert tools links:
For those who are interested in model conversion, you can try to export axmodel through the original repo : 

[How to Convert from ONNX to axmodel](https://github.com/AXERA-TECH/3D-Speaker-MT.axera) 

## 支持平台

- AX650N  

## 功能

会议音频转录

## 模型转换  

参考[模型转换](https://github.com/AXERA-TECH/3D-Speaker-MT.axera/tree/main/model_convert)  


## 上板部署

-  AX650N 的设备已预装 Ubuntu22.04  
-  以 root 权限登陆 AX650N 的板卡设备  
-  链接互联网，确保 AX650N 的设备能正常执行 apt install, pip install 等指令   
-  已验证设备：AX650N DEMO Board  

## Python API 运行

在python3.10(验证)  
Requirements  

```
pip3 install -r requirements.txt
```

## 在开发板运行以下命令

```
支持输入音频文件格式：wav，mp3
```

```
python3 ax_meeting_transc_demo.py --output_dir output_dir --wav_file wav/vad_example.wav
```

运行参数说明:

| 参数名称 | 说明|
|-------|------|
| `--output_dir` | 结果保存路径 |
| `--wav_file`  | 音频路径     |
| `--seq_len`   | ASR输入一致，目前固定132 |

输出保存为txt文件，具体结果如下：

```
Speaker_0: [0.000 63.810] 试错的过程很简单，而且特别是今天报名仓雪卡的同学，你们可以。听到后面的有专门的活动课，他会大大降低你的试绸成本。其实你也可以不来听课。为什么你自己写嘛？我写今天写5个点，我就试试试验一下，反正这5个点不行，我再写5个点，这是再不行。那再写5个点吧，。你总会所谓的活动大神和所谓的高手都是只有一个。把所有的错，所有的坑全国趟一遍，留下正确的你就是所谓的大神。明白吗？所以说关于活动通过这一块，我只送给你们四个字啊，换位思考。如果说你要想降低。你的试错成本，今天来这里你们就是对的。。因为有畅血唱血卡这个机会，所以说关于活动过于不过这个问题，或者活动很难通过这个话题。呃，如果真的。那要坐下来聊的话，要聊一天。但是我觉得我刚才说的四个字足够。好，谢谢。
Speaker_1: [63.810 70.471] 好，非常感谢那个三茂老师的回答啊。三茂老师说我们在。整个店铺的这个活动当中，我们要学会换位思考。其实我。
```

## Latency

AX650N  
| model | latency(ms) |
|------|------|
| vad | `5.441` |
| cammplus | `2.907` |
| sensevoice | `25.482` |

RTF: 约为0.2
```
eg:
Inference time for vad_example.wav: 10.92 seconds
  - VAD processing time: 2.20 seconds
  - Speaker embedding extraction time: 1.88 seconds
  - Speaker clustering time: 0.16 seconds
  - ASR processing time: 3.75 seconds
load model  + Inference time for vad_example.wav: 13.08 seconds
Audio duration: 70.47 seconds
RTF: 0.15
```

参考： 
- [3D-Speaker](https://https://github.com/modelscope/3D-Speaker/tree/main)
- [sensevoice.axera](https://github.com/ml-inory/sensevoice.axera/tree/main)  
- [3D-Speaker.axera](https://github.com/AXERA-TECH/3D-Speaker.axera/tree/master)

## 技术讨论
- Github issues  
- QQ 群: 139953715