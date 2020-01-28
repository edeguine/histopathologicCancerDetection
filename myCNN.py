import torch.nn as nn

class myConvNet(nn.Module):
    def __init__(self):
        super(myConvNet, self).__init__()

        filters = [3, 32, 64, 128]
        self.layers = nn.ModuleList()

        for i in range(3):
            self.layers.append(nn.Sequential(
                nn.Conv2d(filters[i], filters[i + 1], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(filters[i + 1]),
                nn.ReLU(),
                nn.Conv2d(filters[i + 1], filters[i + 1], kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(filters[i + 1]),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout()))

        self.layers.append(nn.Sequential(
            nn.Linear(128 * 12 * 12, 1000),
            nn.Linear(1000, 256),
            nn.Linear(256, 2)))

    def forward(self, x):
        out = self.layers[0](x)
        for i, l in enumerate(self.layers[1:3]):
            out = l(out)
        out = out.view(-1, 128 * 12 * 12)
        out = self.layers[3](out)
        return out

