import torch
from model import AttentionNet
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

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

# You may need to adjust the dataset path and class accordingly
test_dataset = datasets.ImageFolder(root='test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load model
model = AttentionNet().to(device)
model.load_state_dict(torch.load('best_model.pth', map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        # images: [batch, channels, height, width]
        # The AttentionNet expects input shape: [batch, num_frames, channels, height, width]
        # For static images, we can add a dummy num_frames dimension of 1
        images = images.unsqueeze(1).to(device)  # [batch, 1, C, H, W]
        labels = labels.to(device)
        
        # Forward pass through the complete AttentionNet model
        outputs = model(images)  # [batch, num_classes]
        
        # Get predictions
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = 100.0 * correct / total if total > 0 else 0.0
print(f"Test Accuracy: {accuracy:.2f}%")
