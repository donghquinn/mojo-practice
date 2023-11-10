import torch.nn as nn


class CnnModel(nn.Module):
    def __init__(self, input_size: int | tuple[int], kernel_size: int | tuple[int], padding_size: int, stride: int):
        super(CnnModel, self).__init__()

        firstOuputSize = (input_size + 2 * padding_size -
                          kernel_size) / stride + 1

        print("1st Layer output Size: {}".format(
            firstOuputSize))

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_size, out_channels=firstOuputSize,
                      kernel_size=kernel_size, padding=padding_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        secondOutputSize = (firstOuputSize + 2 *
                            padding_size - kernel_size) / stride + 1

        print("2nd Layer output Size: {}".format(
            secondOutputSize))

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=firstOuputSize,
                      out_channels=secondOutputSize, kernel_size=kernel_size, padding=padding_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.fc = nn.Sequential(
            nn.Flatten(secondOutputSize),
            nn.Linear(secondOutputSize),
            nn.Softmax()
        )

    def forward(self, x):
        out = self.layer1(x)

        print("First Layer Convolution Output Size: {}".format(out.size))

        out2 = self.layer2(out)

        print("Second Layer Convolution Output Size: {}".format(out2.size))

        out3 = self.fc(out2)

        return out3
