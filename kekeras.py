import numpy as np
from keras.models import Sequential
from keras.layers import Dense
'''
Реализация обычного XOR через нейросетку ( сори, без пресептронов, они не могут делать XOR)
'''
# Первым делом - создаём дата-сет
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# Включаем "Мозг"
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu')) # Количество измерений и принцип активации
model.add(Dense(1, activation='sigmoid')) # Хотя не , вот вам 1 пресептрон

# Подгоняем модель
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Делаем модель фитоняшкой
model.fit(X_train, y_train, epochs=1000, verbose=0)

# Тестируем
X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_test = np.array([[0], [1], [1], [0]])

loss, accuracy = model.evaluate(X_test, y_test)

print("Потери:", loss)
print("Точность:", accuracy)
