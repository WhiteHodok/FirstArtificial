import torch
import torch.nn as nn
import torch.optim as optim

# ААА ФУНКЦИИ
class XORModel(nn.Module):
    def __init__(self):
        super(XORModel, self).__init__()
        self.fc1 = nn.Linear(2, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# ААА ДАТАСЕТЫ
X_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# ААА ЭТО ТОТ КТО ЗАБЕРЁТ У НАС РАБОТУ
model = XORModel()

# ААА ЭТО ФУНКЦИЯ ПОТЕРЬ И ОПТИМИЗАЦИИ
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# АААА КАК ОНА УБОГО ТРЕНИРУЕТСЯ ПО СРАВНЕНИЮ С КЕРАС ЗАЧЕМ Я УЧИЛ ЭТОГО ТОРЧКА
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

# ААА ТЕСТЫ
X_test = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_test = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
with torch.no_grad():
    output = model(X_test)
    predicted = torch.round(output)
    accuracy = (predicted == y_test).sum().item() / len(y_test)

print("Точность:", accuracy)