import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VideoMAEImageProcessor, VideoMAEModel
import numpy as np
import random

# set the seed for reproduceability
seed = 0
torch.manual_seed(seed)
np.random.seed(0)
random.seed(0)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Define the path to the grey image and the folder containing the video frames
grey_image_path = "./grey_image.png"  # Path to the grey image
video_frames_dir = "./MOVIE_DATA"  # Directory containing video frames

# Load pre-trained VideoMAE model and feature extractor
model_name = "MCG-NJU/videomae-base"  # Replace with the specific model you're using
feature_extractor = VideoMAEImageProcessor.from_pretrained(model_name)
model = VideoMAEModel.from_pretrained(model_name).to(device)

# custom class for loading video clips padded with grey images on either end
class CustomVideoDataset(Dataset):
    def __init__(self, grey_image_path, video_frames_dir, transform=None):
        self.grey_image_path = grey_image_path
        self.video_frames_dir = video_frames_dir
        self.transform = transform
        
# Load and prepare the grey image
        print(f"Loading grey image from: {grey_image_path}")
        self.grey_image = Image.open(grey_image_path).convert("RGB").resize((224, 224))

        # Load video frames and sort based on naming conventions
        self.frame_files = sorted(
            [os.path.join(video_frames_dir, f) for f in os.listdir(video_frames_dir) if f.endswith('.png')],
            key=lambda x: (
                int(''.join(filter(str.isdigit, os.path.basename(x).split('_')[0]))), 
                int(''.join(filter(str.isdigit, os.path.basename(x).split('_')[-1].split('.')[0])))
            )
        )

        self.video_clips = {}
        for frame_file in self.frame_files:
            clip_prefix = '_'.join(os.path.basename(frame_file).split('_')[:2])
            if clip_prefix not in self.video_clips:
                self.video_clips[clip_prefix] = []
            self.video_clips[clip_prefix].append(frame_file)

        self.clip_keys = list(self.video_clips.keys())
        print(f"Found {len(self.clip_keys)} video clips.")

    def __len__(self):
        return len(self.clip_keys)
    
    def __getitem__(self, idx):
        clip_key = self.clip_keys[idx]
        frame_files = self.video_clips[clip_key]
        
        # Construct the sequence of frames
        frames = [self.grey_image] * 16  # 16 frames of the grey image
        frames += [Image.open(frame_file).convert("RGB").resize((224, 224)) for frame_file in frame_files]  # 15 frames of the video clip
        frames += [self.grey_image] * 16  # 16 frames of the grey image
        
        if self.transform:
            frames = self.transform(frames, return_tensors="pt")['pixel_values']

        return frames

    
def collate_fn(batch):
    # Since we only have one sequence per batch, we just return the single batch item
    return batch[0]



num_tubelets = 8 
num_patches = 196


# Create dataset and dataloader
dataset = CustomVideoDataset(grey_image_path, video_frames_dir, transform=feature_extractor)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# Process each clip in sliding window manner
print("Processing batches...")
all_hidden_states = []
total_frames = 0
for batch_idx, frames in enumerate(dataloader):
    model.eval()
    frames = frames.squeeze(0)  # Remove the batch dimension

    # Slide over the frames with a window of 16 (standard input to VideoMAE)
    num_windows = frames.shape[0] - 15
    total_frames += num_windows  # Update total number of windows processed

    for i in range(frames.shape[0] - 15):
        input_frames = frames[i:i+16].unsqueeze(0).to(device)  # Add batch dimension
        inputs = {'pixel_values': input_frames}

        # Run the video through the model
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract hidden states
        hidden_states = outputs.last_hidden_state
        all_hidden_states.append(hidden_states.cpu().numpy())
        print(f"Processed frame window {i + 1}/{frames.shape[0] - 15} for batch {batch_idx + 1}/{len(dataloader)}")

# Convert to numpy array
all_hidden_states = np.concatenate(all_hidden_states, axis=0)  

# Reshape to separate the 8 temporal tublets
all_hidden_states = all_hidden_states.reshape((total_frames, num_tubelets, num_patches, 768))  

# Pool the last section (last tublet)
last_tublet = all_hidden_states[:, -1, :, :]  

# Apply average pooling on the last 196 units for each hidden unit
pooled_vector = np.mean(last_tublet, axis=1)  

# For max pooling:
# pooled_vector = np.max(last_tublet, axis=1)  

# Save the pooled vector to a file
np.save('hidden_states.npy', pooled_vector)

print("Processing and saving complete.")
