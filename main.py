import torch
from model import MattingNetwork
import glob
from inference import convert_video


model = MattingNetwork('mobilenetv3').eval().cuda()
model.load_state_dict(torch.load('user_data/rvm_mobilenetv3.pth'))

motionpaths = glob.glob('xfdata/testA/motion/*')
staticpaths = glob.glob('xfdata/testA/static/*')
all_paths = motionpaths + staticpaths

for pths in all_paths:
    input_source = pths
    output_composition = pths.replace('xfdata/testA', 'prediction_result')
    convert_video(
        model,  # 模型，可以加载到任何设备（cpu 或 cuda）
        input_source=input_source,  # 视频文件，或图片序列文件夹
        output_type='video',  # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
        output_composition=output_composition,  # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
        output_video_mbps=7.5,  # 若导出视频，提供视频码率
        downsample_ratio=0.7,  # 下采样比，可根据具体视频调节，或 None 选择自动
        seq_chunk=1,  # 设置多帧并行计算
    )


