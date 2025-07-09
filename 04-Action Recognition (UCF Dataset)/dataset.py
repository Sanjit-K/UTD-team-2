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
        self.transform = transform
        self.samples = []
        classToIndex = {}
        for i in open(class_index, 'r'):
            index, class_name = i.split()
            classToIndex[class_name] = int(index) - 1
        for i in open(splits, 'r'):
            video_path, label = i.split()
            if label is None:
                class_name = video_path.split('/')[0]
                label = classToIndex[class_name]
            else:
                label = int(label) - 1
            self.samples.append((video_path, label))
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label = self.samples[index]
        video_filename = os.path.basename(video_path)
        video_path = os.path.join('UCF101', video_filename)
        video = cv2.VideoCapture(video_path)
        frames = []
        while True:
            exists, frame = video.read()
            if not exists:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        video.release()
        if len(frames) < 16:
            frames += [frames[-1]] * (16 - len(frames))
            video_indeces = np.linspace(0, len(frames)-1, 16).astype(int)
            frameFixed = [frames[i] for i in video_indeces]
            frameFixed = [self.transform(frame) for frame in frameFixed]
            tensor = torch.stack(frameFixed, dim=1) # RGB channel, Time, height, width
            return tensor, label


