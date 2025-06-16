# ViStream
Official Repo of ViStream Published at CVPR 2025

## Model Checkpoint

The model checkpoint file is hosted on Hugging Face due to its large size (292MB). 

### Download Instructions

You can download the checkpoint file using one of the following methods:

#### Method 1: Using wget/curl
```bash
# Download the checkpoint file
wget https://huggingface.co/AndyBlocker/ViStream/resolve/main/checkpoint-90.pth
```

#### Method 2: Using Hugging Face Hub
```bash
# Install huggingface_hub if not already installed
pip install huggingface_hub

# Download using Python
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='AndyBlocker/ViStream', filename='checkpoint-90.pth', local_dir='.')"
```

#### Method 3: Using Git LFS (after cloning the HF repo)
```bash
# Clone the Hugging Face repository
git clone https://huggingface.co/AndyBlocker/ViStream
# Copy the checkpoint to your project directory
cp ViStream/checkpoint-90.pth ./
```

After downloading, make sure the checkpoint file is placed in the root directory of this project.

## Running Experiments

To run inference experiments, use the `eval.sh` script. The script contains various test commands for different tracking tasks:

- **SOT (Single Object Tracking)**: Uncomment the `test_sot_siamfc.py` or `test_sot_cfnet.py` lines
- **VOS (Video Object Segmentation)**: Uncomment the `test_vos.py` lines  
- **MOT (Multiple Object Tracking)**: Uncomment the `test_mot.py` lines
- **MOTS (Multiple Object Tracking and Segmentation)**: Uncomment the `test_mots.py` lines
- **Pose Tracking**: Uncomment the `test_posetrack.py` lines

**Usage:** Uncomment the desired experiment lines in `eval.sh`, then run:
```bash
bash eval.sh
```

## Acknowledgments

This project is based on [UniTrack](https://github.com/Zhongdao/UniTrack) with improvements for energy-efficient tracking.

The energy consumption and SOP evaluation code is adapted from [syops-counter](https://github.com/iCGY96/syops-counter).
