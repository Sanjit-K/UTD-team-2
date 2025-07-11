import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import argparse

batch_size = 32

def parser():

    args_reader = argparse.ArgumentParser(description="Script to train a linear regression model with two parameteres on house rent datasest.")
    
    # initialize hyper-parameters
    args_reader.add_argument('--lr', type=float, default=1e-4, help='learning rate for training the linear regression model')
    args_reader.add_argument('--epochs', type=int, default=50, help='number of epochs to train the model on.')
    # args_reader.add_argument('--val_interval', type=int, default=25, help='number of updates between each validation') 
    # args_reader.add_argument('--lr_drop_epoch', type=int, default=200, help='Epoch at which lr drops.')
    # args_reader.add_argument('--lr_drop_factor', type=float, default=0.1, help='Factor by which the learning rate drops.' )

    args = args_reader.parse_args()

    return args

if __name__ == "__main__":

    args = parser()
    print('Training with learning rate: ', args.lr)
    print('Epochs: ', args.epochs)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root="03-CIFAR 10 (Convolutional Neural Network)", train=True, transform=transform, download=True)
train, val = torch.utils.data.random_split(train_data, [45000, 5000])
test = torchvision.datasets.CIFAR10(root="03-CIFAR 10 (Convolutional Neural Network)", train=False, transform=transform, download=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")


#TODO: write a custom dataloader
# class Custom_DataLoader(DataLoader):
#     def __init__(self, dataset, batch_size : int, shuffle : bool):
#         if (shuffle==True):
#             data = random.shuffle(dataset)

#         dataset = iter(dataset)
#         print(torch.type(dataset))
#     # def __getitem__(index):

#     # def __len__(self):
#     #     return self.len
# Custom_DataLoader(train,batch_size,True)

train_set = DataLoader(train, batch_size=batch_size, shuffle=True)
val_set = DataLoader(val, batch_size=batch_size, shuffle=True)
test_set = DataLoader(test, batch_size=batch_size, shuffle=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_identity=None, stride=1):
        super(ResBlock, self).__init__()
        self.dropout = nn.Dropout(0.3)

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
        self.downsample_identity = downsample_identity
    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample_identity is not None: # ? 
            identity = self.downsample_identity(identity)

        x += identity
        x = self.relu(x)

        return x
class ResNet(nn.Module):
    def __init__(self, ResBlock, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(in_channels= image_channels, out_channels=16, kernel_size=6, stride=2, padding=2) # [32,3,32,32] --> [32,16,16,16]
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU() 
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2) # [32,16,16,16] --> [32,16,8,8]
    
        # Res-Net Layers
        self.layer1 = self.res_layer(ResBlock, layers[0], out_channels=32, stride=1)
        self.layer2 = self.res_layer(ResBlock, layers[1], out_channels=64, stride=2)
        self.layer3 = self.res_layer(ResBlock, layers[2], out_channels=128, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128,num_classes)
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    def res_layer(self, ResBlock, num_residual_blocks, out_channels, stride):
        downsample_identity = None
        layers = []

        if (stride != 1 or self.in_channels != out_channels): # * 2?
            downsample_identity = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride), 
                nn.BatchNorm2d(out_channels)
            )

        layers.append(ResBlock(self.in_channels, out_channels, downsample_identity, stride))
        self.in_channels = out_channels


        for i in range(num_residual_blocks-1):
            layers.append(ResBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

model = ResNet(ResBlock, [2,2,2,2], image_channels=3, num_classes=10).to(device)




def test_params():
    num_params = 0
    for x in model.parameters():
        num_params += len(torch.flatten(x))

    print(f'{num_params:,} parameters')

    test_img, _ = next(iter(test_set))
    test_img = test_img.to(device)
    print(model(test_img).shape)

test_params()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

class train():
    def __init__(self):
        self.train_losses = []
        self.val_losses = []

        for epoch in range(0,args.epochs):
            self.train_losses.append(self.train_epoch(epoch))
            if (epoch%5==4):
                self.val_losses.append(self.validate_epoch(epoch))
            if (epoch%75==74):
                to_continue = input('continue training? [y]/[n] ')
                if (to_continue.lower()=='n'):
                    break
            scheduler.step()
        self.test()        
    
        plt.figure()

        plt.plot(range(args.epochs), self.train_losses, label = 'Training Loss', color='blue', linestyle='-')
        plt.plot(range(0, args.epochs, 5), self.val_losses, label = 'Validation Loss', color='red', linestyle='-')

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()

        plt.show()
        torch.save(model.state_dict(), "03-CIFAR 10 (Convolutional Neural Network)/model_4.pt")
        print("Saved PyTorch Model State to model.pt")         

    def train_epoch(self, epoch):
        batches_loss = 0
        model.train(True)
        print('---train---')
        global_loss = 0
        for batch_i, (train_imgs, train_labels) in enumerate(train_set):
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            
            # Calc functions
            y_hat = model(train_imgs)
            loss = loss_fn(y_hat, train_labels)
            batches_loss += loss
            global_loss += loss
            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            if (batch_i % 250 == 249):
                print(f'[{epoch+1},{batch_i+1}] Average Loss: {batches_loss/(250)}')
                batches_loss = 0
        return ((global_loss/len(train_set)).item())
    def validate_epoch(self, epoch):
        model.eval()
        batches_loss = 0
        global_loss = 0
        print('---validate---')
        
        with torch.no_grad():
            for batch_i, (val_imgs, val_labels) in enumerate(val_set):
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                y_hat = model(val_imgs)
                loss = loss_fn(y_hat, val_labels)
                batches_loss += loss
                global_loss += loss
                if (batch_i % 50 == 49):
                    print(f'[{epoch+1}, {batch_i+1}] Average Loss: {batches_loss/(50)}')
                    batches_loss = 0
        return ((global_loss/len(val_set)).item())
    def test(self):
        model.eval()
        correct = 0
        total = 0

        print('---test---')
        with torch.no_grad():
            for batch_i, (test_imgs, test_labels) in enumerate(test_set):
                test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)

                y_hat = model(test_imgs)

                _, predicted = torch.max(y_hat, 1)
                total+=test_labels.size(0)
                correct += (predicted==test_labels).sum().item()
        print(f'Accuracy {100 * (correct / total)}%')

train()