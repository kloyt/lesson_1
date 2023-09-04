import torch
import pandas as pd
import torch.nn as nn


class LinearRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss(reduction='mean')

    def fit(self, x, y):
        num_samples, num_features = x.shape
        self.model = nn.Linear(num_features, 1)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for _ in range(self.num_iterations):
            predictions = self.model(x)
            loss = self.criterion(predictions, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict(self, x):
        with torch.no_grad():
            return self.model(x)


if __name__ == '__main__':
    file = pd.read_csv("csv/housing.csv", delimiter=r'\s+')
    data = torch.tensor(file.values, dtype=torch.float32)
    train_x = data[0:255, 0:13]
    train_y = data[0:255, 13].view(-1, 1)
    test_x = data[255:, 0:13]
    test_y = data[255:, 13].view(-1, 1)

    model = LinearRegression(learning_rate=0.01, num_iterations=1000)
    model.fit(train_x, train_y)

    test_predictions = model.predict(test_x)
    print("Test Predictions:", test_predictions)