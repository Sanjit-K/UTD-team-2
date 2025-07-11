import torch
from model import AttentionNet
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import AttentionNet
from data import UCFdataset
import multiprocessing


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define test dataset and dataloader
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_index_file = r'C:\Users\dylan\UTD-team-2\ucfTrainTestlist\classInd.txt'
train_split = r"C:\Users\dylan\UTD-team-2\ucfTrainTestlist\trainlist01.txt"
test_split = r'C:\Users\dylan\UTD-team-2\ucfTrainTestlist\testlist01.txt'

# You may need to adjust the dataset path and class accordingly
test_dataset = UCFdataset(class_index_file, test_split, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

# Load model
model = AttentionNet().to(device)
# model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

total_loss = 0
correct = 0
total = 0

with torch.no_grad():
    for videos, labels in enumerate(test_loader):
        
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        
        _, predicted = outputs.max(outputs, 1)
        correct += (predicted==labels).sum().item()
        total += labels.size(0)
            
    
accuracy = 100.0 * correct / total if total > 0 else 0.0
print(f"Test Accuracy: {accuracy:.2f}%")
