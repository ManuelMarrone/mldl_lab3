import torch
from torch import nn

# Define the custom neural network
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Define layers of the neural network
        #input: TinyImageNet 64x64x3
        #usa blocchi ripetibili che puoi scalare
        #kernel 3x3 è un buon compromesso
        # usa stride = 1 epadding = 1 per mantenre la dimensione spaziale
        # dopo conv usa BatchNorm
        #primo layer 64 filtri perchè deve imparare feature di base
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        #secondo layer per normalizzare
        self.bn1 = nn.BatchNorm2d(64) #64 è il numero di canali in uscita da conv
        #terzo layer introdurre non linearità
        self.relu1 = nn.ReLU()
        #quarto layer conv che deve prendere pattern più complessi
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.poolmax = nn.MaxPool2d(2,2)
        #quinto layer
        self.bn2 = nn.BatchNorm2d(128)
        #sesto layer
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        #maxpool per dimezzare la dimensionme spaziale
        self.pool1 = nn.AdaptiveAvgPool2d( (1,1) )
        self.flatten = nn.Flatten()
        #fc1 deve ricevere un tensore [B,256]
        self.fc1 = nn.Linear(256, 200) # 200 is the number of classes in TinyImageNet

    def forward(self, x):
       # Primo blocco conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Secondo blocco conv
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.poolmax(x)  # aggiungi qui il maxpool

        # Terzo blocco conv
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # Pooling adattivo
        x = self.pool1(x)

        # Flatten
        x = self.flatten(x)

        # Fully connected
        x = self.fc1(x)

        return x
