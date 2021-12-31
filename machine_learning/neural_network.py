from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import torch.utils.data as utils

import matplotlib.pyplot as plt

# The parts that you should complete are designated as TODO
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        # TODO: define the layers of the network
        self.hidden_1 = nn.Linear(784, 64)
        self.hidden_2 = nn.Linear(64, 32)

    def forward(self, x):
        # TODO: define the forward pass of the network using the layers you defined in constructor
        h1 = F.relu(self.hidden_1(x))
        h2 = F.relu(self.hidden_2(h1))
        output = F.softmax(h2)
        
        return output
        
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0: #Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
    accuracy = test(model, device, train_loader)
    return accuracy

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def main():
    torch.manual_seed(1)
    np.random.seed(1)
    # Training settings
    use_cuda = False # Switch to False if you only want to use your CPU
    learning_rate = 0.002
    NumEpochs = 15
    batch_size = 32

    device = torch.device("cuda" if use_cuda else "cpu")

    train_X = np.load('../../Data/mnist/X_train.npy')
    train_Y = np.load('../../Data/mnist/y_train.npy')

    test_X = np.load('../../Data/mnist/X_test.npy')
    test_Y = np.load('../../Data/mnist/y_test.npy')

#   train_X = train_X.reshape([-1,1,28,28]) # the data is flatten so we reshape it here to get to the original dimensions of images
#   test_X = test_X.reshape([-1,1,28,28])

    # transform to torch tensors
    tensor_x = torch.tensor(train_X, device=device)
    tensor_y = torch.tensor(train_Y, dtype=torch.long, device=device)

    test_tensor_x = torch.tensor(test_X, device=device)
    test_tensor_y = torch.tensor(test_Y, dtype=torch.long)

    train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

    test_dataset = utils.TensorDataset(test_tensor_x,test_tensor_y) # create your datset
    test_loader = utils.DataLoader(test_dataset) # create your dataloader if you get a error when loading test data you can set a batch_size here as well like train_dataloader

    model = NNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #initialize a list to save training accuracy
    training_accuracy = []
    #initialize a list to save testing accuracy
    testing_accuracy = []
    
    
    for epoch in range(NumEpochs):
        train_acc = train(model, device, train_loader, optimizer, epoch)
        #append training accuracy list
        training_accuracy.append(train_acc)
        print('\nTrain set Accuracy: {:.0f}%\n'.format(train_acc))
        test_acc = test(model, device, test_loader)
        #append testing accuracy list
        testing_accuracy.append(test_acc)
        print('\nTest set Accuracy: {:.0f}%\n'.format(test_acc))

    torch.save(model.state_dict(), "mnist_nn.pt")

    #TODO: Plot train and test accuracy vs epoch
    plt.plot(list(range(1,16)), training_accuracy, label='Training Accuracy')
    plt.plot(list(range(1,16)), testing_accuracy, label='Testing Accuracy')
    #plt.xticks(np.arange(list(range(1,16))))
    plt.title('Training and Testing Accuracy for\nIncreasing Number of Epochs')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('../Figures/nn_accuracy.png')
    plt.clf()
    

if __name__ == '__main__':
    main()
