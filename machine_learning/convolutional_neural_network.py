from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import numpy as np
import torch.utils.data as utils

#for checking summary of model
from torchvision import models
from torchsummary import summary

#for plotting
import matplotlib.pyplot as plt

# The parts that you should complete are designated as TODO
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # TODO: define the layers of the network
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout2d(p=0.25)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(128,10)
        
    def forward(self, x):
        # TODO: define the forward pass of the network using the layers you defined in constructor
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        #accuracy = 100. * batch_idx / len(train_loader)
        if batch_idx % 100 == 0: #Print loss every 100 batch
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                loss.item()))
    print('-'*20)
    print('Training Accuracy (not test accuracy):')
    accuracy = test(model, device, train_loader)
    print('-'*20)
    
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
    print('\nTest set Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        accuracy))
    print('-'*20)
    
    return accuracy


def main():
    print('-'*20)
    print('Start of CNN Script')
    torch.manual_seed(1)
    np.random.seed(1)
    # Training settings
    use_cuda = False # Switch to False if you only want to use your CPU
    learning_rate = 0.01
    NumEpochs = 10
    batch_size = 32

    device = torch.device("cuda" if use_cuda else "cpu")
    
    train_X = np.load('../../Data/X_train.npy')
    train_Y = np.load('../../Data/y_train.npy')

    test_X = np.load('../../Data/X_test.npy')
    test_Y = np.load('../../Data/y_test.npy')
    
    train_X = train_X.reshape([-1,1,28,28]) # the data is flatten so we reshape it here to get to the original dimensions of images
    test_X = test_X.reshape([-1,1,28,28])

    # transform to torch tensors
    tensor_x = torch.tensor(train_X, device=device) 
    tensor_y = torch.tensor(train_Y, dtype=torch.long, device=device)
    
    test_tensor_x = torch.tensor(test_X, device=device)
    test_tensor_y = torch.tensor(test_Y, dtype=torch.long)

    train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size) # create your dataloader
    
    test_dataset = utils.TensorDataset(test_tensor_x,test_tensor_y) # create your datset
    test_loader = utils.DataLoader(test_dataset) # create your dataloader if you get a error when loading test data you can set a batch_size here as well like train_dataloader
    
    model = ConvNet().to(device)
    #print a summary of layer dimensions and parameters
    print('Model Summary:')
    summary(model, input_size=(1,28,28))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #create lists to store training and testing accuracy for plotting
    training_accuracy = []
    testing_accuracy = []
    
    for epoch in range(NumEpochs):
        train_acc = train(model, device, train_loader, optimizer, epoch)
        #append training accuracy list
        training_accuracy.append(train_acc)
        test_acc = test(model, device, test_loader)
        #append testing accuracy list
        testing_accuracy.append(test_acc)
        
    torch.save(model.state_dict(), "mnist_cnn.pt")
    
    #TODO: Plot train and test accuracy vs epoch
    plt.plot(list(range(1,11)), training_accuracy, label='Training Accuracy')
    plt.plot(list(range(1,11)), testing_accuracy, label='Testing Accuracy')
    plt.title('Training and Testing Accuracy\nfor Each Epoch')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('../Figures/cnn_accuracy.png')
    plt.show()
    plt.clf()
    
    print('End of CNN Script')
    print('-'*20)
    print('End of HW5 Scripts')


if __name__ == '__main__':
    main()


#Below was used to generate the predictions for the submission folder
#select all and run selection if you would like to reproduce
#digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#model.eval()
#correct = 0
#predictions_vect = []
#with torch.no_grad():
#    for data, target in test_loader:
#        data, target = data.to(device), target.to(device)
#        output = model(data)
#        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
#        correct += pred.eq(target.view_as(pred)).sum().item()
#        _, predictions = torch.max(output, 1)
#        predictions_vect.append(digits[predictions])   
#np.savetxt('../Predictions/predictions_MNIST.csv', predictions_vect, delimiter=',')