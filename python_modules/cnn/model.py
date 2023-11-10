import torch.nn as nn


class CnnModel(nn.Module):
    def __init__(self):
        super(CnnModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, kernel_size=5, padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(),
            nn.ReLU(),
            nn.MaxPool2d(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(),
            nn.Softmax()
        )

    def forward(self, x):
        out, _ = self.layer1(x)

        print("First Layer Size: {}".format(out.size))
        out2, _ = self.layer2(out)

        out3, _ = self.fc(out2)

        return out3
