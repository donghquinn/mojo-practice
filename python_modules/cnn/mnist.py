from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_mnist_dataset(batch_size: int):
    transform = transforms.ToTensor()

    train_data = datasets.MNIST(
        root='mnist_train', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(
        root='mnist_test', train=False, transform=transform, download=True)

    train_tensor = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_tensor = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    print("Train Tensor Shape: {}".format(train_tensor))
    print("Test Data Shape: {}".format(test_data))

    return train_data, test_data, train_tensor, test_tensor
