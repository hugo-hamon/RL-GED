import torch.nn.functional as F
import torch.nn as nn
import torch


class CNNModel(nn.Module):

    def __init__(self, state_size: int, input_size: int, num_hidden: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            input_size, num_hidden, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=3, padding=1
        )
        self.fc = nn.Linear(num_hidden * state_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through thenetwork."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ResBlock(nn.Module):

    def __init__(self, num_hidden: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, state_size: int, input_size: int, num_resBlocks: int, num_hidden: int) -> None:
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(input_size, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )

        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for _ in range(num_resBlocks)]
        )

        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * state_size, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        return self.valueHead(x)


if __name__ == "__main__":
    size = 2
    model = CNNModel(size**2, 2, 4)
    print(model)
    print(model(torch.rand(1, 2, size, size)).shape)

    model = ResNet(size**2, 2, 4, 4)
    print(model)
    print(model(torch.rand(1, 2, size, size)).shape)
