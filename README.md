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
