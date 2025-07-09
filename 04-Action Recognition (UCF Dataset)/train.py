from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from model import AttentionNet
from data import UCFdataset
import os 

epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")

frame_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(128),
    transforms.CenterCrop(112),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_index_file = r'/Users/dylanjoseph/Library/CloudStorage/OneDrive-Personal/Code/CVMC Lab Work/UTD-team-2/ucfTrainTestlist/classInd.txt'
train_split = r'/Users/dylanjoseph/Library/CloudStorage/OneDrive-Personal/Code/CVMC Lab Work/UTD-team-2/ucfTrainTestlist/trainlist01.txt'
test_split = r'/Users/dylanjoseph/Library/CloudStorage/OneDrive-Personal/Code/CVMC Lab Work/UTD-team-2/ucfTrainTestlist/testlist01.txt'

train_dataset = UCFdataset(class_index_file, train_split, transform=frame_transform)
test_dataset = UCFdataset(class_index_file, test_split, transform=frame_transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=1)

def train(model, loader, criterion, optimizer):
    print('training')
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (videos, labels) in enumerate(loader):
        print(f"batch {batch_idx+1}/{len(loader)}")
        videos, labels = videos.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    
    avg_loss = total_loss / total
    acc = correct / total
    print(f"avg_loss={avg_loss:.4f}, acc={acc:.4f}")
    return avg_loss, acc

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in loader:
            
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    avg_loss = total_loss / total
    acc = correct / total
    return avg_loss, acc

if __name__ == "__main__":

    model = AttentionNet()
    model = model.to(device)

    # # Load best_model.pth as starting parameters if it exists
    # if os.path.exists('best_model.pth'):
    #     model.load_state_dict(torch.load('best_model.pth', map_location=device))
    #     print("Loaded best_model.pth as starting parameters.")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_acc = 0

    print('starting')

    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f'on epoch {epoch}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}')
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')