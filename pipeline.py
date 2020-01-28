import torch
import torch.nn as nn
import torch.nn.functional as F

import loadData
import numpy as np

from myCNN import myConvNet

cuda = True 

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

def main():
    model = myConvNet()

    if cuda:
        model = model.cuda()
    model.apply(weights_init)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    
    mean_train_losses = []
    epochs = 10000

    trainref = 'bigtrain'
    testref = 'bigtest'
    print(f"Trainref: {trainref}, testref: {testref}")
    trainset = loadData.loadDataset(trainref, flipHorizontal=True, flipVertical=True, meanNorm=True, stdNorm=False)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=500, shuffle=True, num_workers=2)
    testset = loadData.loadDataset(testref, flipHorizontal=False, flipVertical=False, meanNorm=True, stdNorm=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=12, shuffle=True, num_workers=2)

    train = True 

    for epoch in range(epochs):
        if train:
            train_losses = []
            for i, (images, labels) in enumerate(train_loader):

                optimizer.zero_grad()
                inputs = images.float()

                if cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                outputs = model(inputs)
                
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                    
            mean_train_losses.append(np.mean(train_losses))
            print("Train losses: ", np.mean(train_losses))

        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                inputs = images.float()
                if cuda:
                    inputs = inputs.cuda()
                    labels = labels.cuda()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        if epoch % 10 == 0:
            # Save model
            torch.save(model.state_dict(), f"models/model_{epoch}.pth")
        accuracy = 100 * correct / total
        print('epoch : {}, train loss : {:.4f} accuracy: {:.4f}'.format(epoch + 1, np.mean(train_losses), accuracy))

if __name__ == "__main__":
    main()
