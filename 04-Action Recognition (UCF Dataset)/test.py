import torch
from model import AttentionNet
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import UCFdataset
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Test Action Recognition Model")
    parser.add_argument('--class_index_file', type=str, default="ucfTrainTestlist/classInd.txt", help='Path to classInd.txt')
    parser.add_argument('--test_split', type=str, default="ucfTrainTestlist/testlist01.txt", help='Path to testlist01.txt')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of DataLoader workers')
    parser.add_argument('--mha_layers', type=int, default=2, help='Number of MHA layers in AttentionNet')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads in MHA')
    parser.add_argument('--model_load_path', type=str, default='best_model.pth', help='Path to load the model from')
    parser.add_argument('--resnet_model', type=str, default='resnet18', help='ResNet backbone to use (e.g., resnet18, resnet34)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    return parser.parse_args()

def get_test_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if __name__ == "__main__":
    args = get_args()
    device = torch.device(args.device)
    transform = get_test_transform()

    test_dataset = UCFdataset(args.class_index_file, args.test_split, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = AttentionNet(mha_layers=args.mha_layers, num_heads=args.num_heads, resnet_model=args.resnet_model).to(device)
    if args.model_load_path and torch.cuda.is_available():
        model.load_state_dict(torch.load(args.model_load_path, map_location=device))
        print(f"Loaded model from {args.model_load_path}")
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy:.2f}%")
