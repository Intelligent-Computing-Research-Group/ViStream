#!/usr/bin/env python3
"""
Script to upload checkpoint file to Hugging Face Hub
Usage: python upload_checkpoint.py
"""

import os
from huggingface_hub import HfApi, create_repo

def upload_checkpoint():
    # Configuration
    repo_id = "AndyBlocker/ViStream"
    checkpoint_file = "checkpoint-90.pth"
    
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_file):
        print(f"Error: {checkpoint_file} not found in current directory")
        return False
    
    # Initialize Hugging Face API
    api = HfApi()
    
    try:
        # Create repository if it doesn't exist
        try:
            create_repo(repo_id, exist_ok=True)
            print(f"Repository {repo_id} is ready")
        except Exception as e:
            print(f"Repository creation info: {e}")
        
        # Upload the checkpoint file
        print(f"Uploading {checkpoint_file} to {repo_id}...")
        api.upload_file(
            path_or_fileobj=checkpoint_file,
            path_in_repo=checkpoint_file,
            repo_id=repo_id,
            commit_message=f"Upload {checkpoint_file} - UniTrack model checkpoint"
        )
        
        print(f"✅ Successfully uploaded {checkpoint_file} to https://huggingface.co/{repo_id}")
        print(f"📥 Download URL: https://huggingface.co/{repo_id}/resolve/main/{checkpoint_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error uploading file: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting checkpoint upload to Hugging Face...")
    print("Make sure you have logged in with: huggingface-cli login")
    
    success = upload_checkpoint()
    if success:
        print("\n🎉 Upload completed successfully!")
        print("You can now use the download instructions in README.md")
    else:
        print("\n💥 Upload failed. Please check your credentials and try again.")