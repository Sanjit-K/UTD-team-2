import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torch import nn
from torch.utils.data import DataLoader

batch_size = 64

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root="03-CIFAR 10 (Convolutional Neural Network)", train=True, transform=transform, download=True)
train, val = torch.utils.data.random_split(train_data, [45000, 5000])
test = torchvision.datasets.CIFAR10(root="03-CIFAR 10 (Convolutional Neural Network)", train=False, transform=transform, download=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = DataLoader(train, batch_size=batch_size, shuffle=True)
val_set = DataLoader(val, batch_size=batch_size, shuffle=False)
test_set = DataLoader(test, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(in_features=128, out_features=10)
        )
    def forward(self,x):
        x = self.model(x)
        
        return x

model = CNN().to(device)


def test_params():
    num_params = 0
    for x in model.parameters():
        num_params += len(torch.flatten(x))

    print(f'{num_params:,} parameters')

    test_img, _ = next(iter(test_set))
    test_img = test_img.to(device)
    print(model(test_img).shape)

test_params()

lr = 1e-2
epochs = 50
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

class train():
    def __init__(self):
        self.train_losses = []

        self.val_losses = []
        self.val_epochs = []

        for epoch in range(0, epochs):
            self.train_losses.append(self.train_epoch(epoch))
            if (epoch%5==4 or epoch==0):
                self.val_losses.append(self.validate_epoch(epoch))
                self.val_epochs.append(epoch+1)
            # if (epoch%75==74):
            #     to_continue = input('continue training? [y]/[n] ')
            #     if (to_continue.lower()=='n'):
            #         break
            scheduler.step()
        self.test()        
    
        plt.figure()
        
        plt.plot(range(1,epochs+1), self.train_losses, label = 'Training Loss', color='blue', linestyle='-')
        plt.plot(self.val_epochs, self.val_losses, label = 'Validation Loss', color='red', linestyle='-') #all values shifted to the left :(

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()

        plt.show()
        torch.save(model.state_dict(), "vanilla_cnn.pt")
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