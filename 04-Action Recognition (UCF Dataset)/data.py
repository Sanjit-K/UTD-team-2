import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import cv2


class UCFdataset(Dataset):
    def __init__(self, class_index, splits, transform):
        # Load class name to index mapping
        self.classToIndex = {}
        with open(class_index, 'r') as f:
            for line in f:
                index, class_name = line.strip().split()
                self.classToIndex[class_name] = int(index) - 1
        
        self.transform = transform
        self.samples = []
        classToIndex = {}
        for i in open(class_index, 'r'):
            index, class_name = i.split()
            classToIndex[class_name] = int(index) - 1
        with open(splits, 'r') as f:
            for line in f: 
                parts = line.strip().split()
                video_path = parts[0]
                if len(parts) == 2:
                    label = int(parts[1]) - 1
                else:
                    # Infer label from folder name
                    class_name = video_path.split('/')[0]
                    label = self.classToIndex[class_name]

                self.samples.append((video_path, label))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label = self.samples[index]

        video_filename = os.path.basename(video_path)
        class_name = video_path.split('/')[0]
        main_dir = r"C:\Users\dylan\UTD-team-2\UCF101"
        video_path = os.path.join(main_dir, 'UCF-101', class_name, video_filename)     # use original  relative path, not just basename

        video = cv2.VideoCapture(video_path)
        frames = []
        while True:
            exists, frame = video.read()
            if not exists:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        video.release()
        if len(frames) == 0:
            print("empty video")
        if len(frames) < 16:
            frames += [frames[-1]] * (16 - len(frames))
        video_indices = np.linspace(0, len(frames)-1, 16).astype(int)
        frameFixed = [frames[i] for i in video_indices]
        frameFixed = [self.transform(frame) for frame in frameFixed]
        tensor = torch.stack(frameFixed, dim=0) # Time, RGB channel, height, width
        return tensor, label

