import torch
from model import CnnModel
from mnist import load_mnist_dataset
from sklearn.model_selection import train_test_split


def train():
    learning_rate = 0.01
    epoch = 10
    batch = 50

    train_data, test_data, train_tensor, test_tensor = load_mnist_dataset(
        batch_size=batch)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Available Resource: {}".format(device))

    model = CnnModel(input_size=28, kernel_size=3,
                     padding_size=1, stride=1).to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for i in range(epoch):
        for image, label in next(iter(train_tensor)):
            # 기울기 초기화
            optimizer.zero_grad()

            # 합성곱 연산
            output = model(image)
            # 비용 함수 계산
            cost = criterion(output, test_tensor)
            # 기울기 계산
            cost.backward()
            # 가중치 및 편향 업데이트
            optimizer.step()

        print("[Epoch: {}] cost: {}".format(i, cost))

    model.eval()


train()
