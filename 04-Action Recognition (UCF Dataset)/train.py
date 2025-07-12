from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
from model import AttentionNet
from data import UCFdataset
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Train Action Recognition Model")
    parser.add_argument('--class_index_file', type=str, default="ucfTrainTestlist/classInd.txt", help='Path to classInd.txt')
    parser.add_argument('--train_split', type=str, default="ucfTrainTestlist/trainlist01.txt", help='Path to trainlist01.txt')
    parser.add_argument('--test_split', type=str, default="ucfTrainTestlist/testlist01.txt", help='Path to testlist01.txt')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of DataLoader workers')
    parser.add_argument('--mha_layers', type=int, default=2, help='Number of MHA layers in AttentionNet')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads in MHA')
    parser.add_argument('--model_save_path', type=str, default='best_model.pth', help='Path to save the best model')
    parser.add_argument('--model_load_path', type=str, default='best_model.pth', help='Path to load the model from')
    parser.add_argument('--resnet_model', type=str, default='resnet18', help='ResNet backbone to use (e.g., resnet18, resnet34)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    return parser.parse_args()


def get_frame_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.5, hue=0.1),
        transforms.ToTensor(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.RandomErasing(p=0.2)
    ])

def train(model, loader, criterion, optimizer, device):
    print('training')
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch_idx, (videos, labels) in enumerate(loader):
        print(f'Training batch {batch_idx+1}/{len(loader)}')
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

        if batch_idx%100==99 or batch_idx==0:
            print(f'[{batch_idx+1}, {loss}]')
    
    avg_loss = total_loss / total
    
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy

def evaluate(model, loader, criterion, device):
    print("Evaluation started")
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (videos, labels) in enumerate(loader):
            
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    
    return avg_loss, accuracy

if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device)
    frame_transform = get_frame_transform()

    train_dataset = UCFdataset(args.class_index_file, args.train_split, transform=frame_transform)
    test_dataset = UCFdataset(args.class_index_file, args.test_split, transform=frame_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = AttentionNet(mha_layers=args.mha_layers, num_heads=args.num_heads, resnet_model=args.resnet_model)
    model = model.to(device)

    # Load model if exists
    if os.path.exists(args.model_load_path):
        model.load_state_dict(torch.load(args.model_load_path, map_location=device))
        print(f"Loaded {args.model_load_path} as starting parameters.")

    criterion = nn.CrossEntropyLoss()
    lr = 5e-5
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9,0.98), weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    train_losses = []
    test_losses = []
    best_acc = 0

    print('starting')

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1} with learning rate {lr}')
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        print(f'on epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}%, test_loss={test_loss:.4f}, test_acc={test_acc:.4f}%')
        test_losses.append(test_loss)
        train_losses.append(train_loss)

        scheduler.step()
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), args.model_save_path)
            print('best weights saved!')
    
    print(f'train losses: {train_losses}')
    print(f'test losses: {test_losses}')
