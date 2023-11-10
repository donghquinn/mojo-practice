from torchvision import datasets, transforms


def load_mnist_dataset():
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root='mnist_train', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(
        root='mnist_test', train=False, transform=transform, download=True)

    print("Train Data Shape: {}".format(train_data))
    print("Test Data Shape: {}".format(test_data))

    return train_data, test_data
