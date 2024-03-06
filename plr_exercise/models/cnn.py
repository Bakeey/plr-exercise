from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) model for classification tasks.

    This network is structured for processing images. It consists of two convolutional layers
    followed by two fully connected layers. Dropout is applied after the first two layers
    for regularization.

    The network expects input images of size 28x28 pixels with a single channel (e.g., grayscale images from MNIST dataset).

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer with 32 filters.
        conv2 (nn.Conv2d): Second convolutional layer with 64 filters.
        dropout1 (nn.Dropout): Dropout layer with a dropout rate of 0.25.
        dropout2 (nn.Dropout): Dropout layer with a dropout rate of 0.5.
        fc1 (nn.Linear): First fully connected layer.
        fc2 (nn.Linear): Second fully connected layer, producing output for 10 classes.
    """
    def __init__(self):
        """
        Initializes the CNN model components and layers.
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Defines the forward pass of the CNN.

        Args:
            x (torch.Tensor): The input tensor containing batch of images.

        Returns:
            torch.Tensor: The output tensor after processing input through the CNN layers
                          and applying log softmax on the final layer for classification.
        """
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
