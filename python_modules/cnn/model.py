import torch.nn as nn


class CnnModel(nn.Module):
    def __init__(self, input_channel: int | tuple[int], output_channel: int | tuple[int], kernel_size: int | tuple[int], padding_size: int, stride: int):
        super(CnnModel, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel,
                      kernel_size=kernel_size, padding=padding_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=output_channel,
                      out_channels=output_channel * 2, kernel_size=kernel_size, padding=padding_size, stride=stride),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.fc = nn.Linear(7 * 7 * 64 * 2, 10),
        # nn.Sequential(
        #     # nn.Flatten(),

        #     nn.Softmax()
        # )

    def forward(self, x):
        out = self.layer1(x)

        print("First Layer Convolution Output Size: {}".format(out.size))

        out2 = self.layer2(out)

        print("Second Layer Convolution Output Size: {}".format(out2.size))

        out2 = out2.view(out2.size(0), -1)

        out3 = self.fc(out2)

        return out3
