import torch
from model import CnnModel
from mnist import load_mnist_dataset
from sklearn.model_selection import train_test_split


def main():
    train_data, test_data = load_mnist_dataset()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(device)

    model = CnnModel(input_size=28, kernel_size=3,
                     padding_size=1, stride=1).to(device)

    learning_rate = 0.01
    epoch = 10
    batch = 10

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    for i in range(epoch):
        # 기울기 초기화
        optimizer.zero_grad()

        # 합성곱 연산
        output = model(train_data)
        # 비용 함수 계산
        cost = criterion(output, test_data)
        # 기울기 계산
        cost.backward()
        # 가중치 및 편향 업데이트
        optimizer.step()

        print("[Epoch: {}] cost: {}".format(i, cost))


main()
