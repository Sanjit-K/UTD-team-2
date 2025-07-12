# accuracy: 80.46%

import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import argparse, time

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

mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize(size = (224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# res_net = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT, progress = True)
# res_net = res_net.to(device)
# res_net = res_net.parameters().to(device)
# weights = torchvision.models.ResNet18_Weights.DEFAULT

# print(res_net.eval())

train_data = torchvision.datasets.CIFAR10(root="03-CIFAR 10 (Convolutional Neural Network)", train=True, transform=transform, download=True)
train, val = torch.utils.data.random_split(train_data, [45000, 5000])
test = torchvision.datasets.CIFAR10(root="03-CIFAR 10 (Convolutional Neural Network)", train=False, transform=transform, download=True)



train_set = DataLoader(train, batch_size=batch_size, shuffle=True)
val_set = DataLoader(val, batch_size=batch_size, shuffle=True)
test_set = DataLoader(test, batch_size=batch_size, shuffle=True)


class modifiedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(weights = torchvision.models.ResNet18_Weights.DEFAULT, progress = True)

        # replace res-net last layer
        self.model.fc = nn.Identity()
        self.model.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.model(x)

        return x
        
clf = modifiedResNet().to(device)

print(clf.eval())


def test_params():
    num_params = 0
    for x in clf.parameters():
        num_params += len(torch.flatten(x))

    print(f'{num_params:,} parameters')

    test_img, _ = next(iter(test_set))
    test_img = test_img.to(device)
    print(clf(test_img).shape)

test_params()

lr = 1e-3
epochs = 50
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(clf.model.fc.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5)
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
            scheduler.step()

            if (epoch%5==4):
                to_continue = input('continue training? [y]/[n] ')
                if (to_continue.lower()=='n'):
                    break
        self.test()        
    
        plt.figure()
        
        plt.plot(range(1,epochs+1), self.train_losses, label = 'Training Loss', color='blue', linestyle='-')
        plt.plot(self.val_epochs, self.val_losses, label = 'Validation Loss', color='red', linestyle='-') #all values shifted to the left :(

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()

        plt.show()
        torch.save(clf.state_dict(), "pre_trained_res_net.pt")
        print("Saved PyTorch Model State to model.pt")        

    def train_epoch(self, epoch):
        batches_loss = 0
        clf.train(True)
        print('---train---')
        global_loss = 0
        for batch_i, (train_imgs, train_labels) in enumerate(train_set):
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            
            # Calc functions
            y_hat = clf(train_imgs)
            loss = loss_fn(y_hat, train_labels)
            batches_loss += loss.item()
            global_loss += loss.item()
            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            if (batch_i % 250 == 249):
                print(f'[{epoch+1},{batch_i+1}] Average Loss: {batches_loss/(250)}')
                batches_loss = 0
        return ((global_loss/len(train_set)))
    def validate_epoch(self, epoch):
        clf.eval()
        batches_loss = 0
        global_loss = 0
        print('---validate---')
        
        with torch.no_grad():
            for batch_i, (val_imgs, val_labels) in enumerate(val_set):
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                y_hat = clf(val_imgs)
                loss = loss_fn(y_hat, val_labels)
                batches_loss += loss.item()
                global_loss += loss.item()
                if (batch_i % 50 == 49):
                    print(f'[{epoch+1}, {batch_i+1}] Average Loss: {batches_loss/(50)}')
                    batches_loss = 0
        return ((global_loss/len(val_set)))
    def test(self):
        clf.eval()
        correct = 0
        total = 0

        print('---test---')
        with torch.no_grad():
            for batch_i, (test_imgs, test_labels) in enumerate(test_set):
                test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)

                y_hat = clf(test_imgs)

                _, predicted = torch.max(y_hat, 1)
                total+=test_labels.size(0)
                correct += (predicted==test_labels).sum().item()
        print(f'Accuracy {100 * (correct / total)}%')

train()