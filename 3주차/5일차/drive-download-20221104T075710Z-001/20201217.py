import keras as kr
import numpy as np
import matplotlib.pyplot as plt
x_train = np.array( [1,2,3,4,5] )
y_train = np.array( [1,2,3,4,5] )

model = kr.models.Sequential()
print(type(model))
model.add(kr.layers.Dense(1, input_dim=1, activation='linear'))
model.summary()
sgd = kr.optimizers.SGD(lr=0.01)
model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=1, epochs=10, shuffle=False)
plt.plot(x_train, y_train, 'bo')
plt.plot(x_train, model.predict(x_train), 'r-')
plt.show()