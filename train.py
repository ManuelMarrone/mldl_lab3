import wandb
from models.custom_model import CustomNet
from utils.download_dataset import download_tiny_imagenet
from data.data_loader import get_dataloaders
import torch
import torch.nn as nn
import torch.optim as optim


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        # Reset gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)  #prediction

        #COMPUTE LOSS
        loss = criterion(outputs, targets)

        #Backward pass
        loss.backward()

        #Update weights
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')



if __name__ == "__main__":
    download_tiny_imagenet()
    train_loader, val_loader = get_dataloaders()

    wandb.init(project="Lab03")
    config = wandb.config
    config.epochs = 10

    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)

        wandb.log({"epoch": epoch})




