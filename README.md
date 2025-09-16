# Video Matting (RVM) — Competition Baseline

## Environment

- Python == 3.8  
- torch == 1.9.0  
- torchvision == 0.10.0  
- tensorboard == 2.5.0  
- pims == 0.5  
- tqdm == 4.61.1  

> Tip: Use a clean virtual environment (conda or venv).  
> Example:
> ```bash
> conda create -n rvm python=3.8 -y
> conda activate rvm
> pip install torch==1.9.0 torchvision==0.10.0 tensorboard==2.5.0 pims==0.5 tqdm==4.61.1
> ```

## Runtime

Approx. **15 minutes** (depends on hardware and video length).

## How to Run

1. Place the main video data in:
/xfdata

2. Place the pretrained model checkpoint:
/user_data/rvm_mobilenetv3.pth

3. Run:
```bash
python main.py


## Model Design

Based on the Robust Video Matting (RVM) project.
Adjusted parameters to fit the dataset and runtime constraints.

## Key Tuning Steps

Downsample ratio tuned to balance runtime efficiency and matting quality.

Temporal settings (state reuse across frames) optimized for stable results.

Post-processing applied for smoother alpha mattes and reduced flicker.

I/O pipeline simplified to support batch video processing from /xfdata.

## Reproducibility

To ensure reproducibility, set a fixed random seed inside main.py.

For logging and visualization, run:

tensorboard --logdir runs

## Acknowledgements

This implementation is adapted from Robust Video Matting (RVM). Please refer to the original repository for full details and citation.
