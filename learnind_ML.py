import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
x= np.array([[50, 1], [55, 1], [60, 1], [65, 2], [70, 2], [75, 2], [80, 2], [85, 2], [90, 2], [95, 2],
    [100, 2], [105, 3], [110, 3], [115, 3], [120, 3], [125, 3], [130, 3], [135, 3], [140, 3], [145, 3],
    [150, 3], [155, 4], [160, 4], [165, 4], [170, 4], [175, 4], [180, 4], [185, 4], [190, 4], [195, 4],
    [200, 4], [210, 5], [220, 5], [230, 5], [240, 5], [250, 5], [260, 5], [270, 5], [280, 5], [290, 5],
    [300, 6], [310, 6], [320, 6], [330, 6], [340, 6], [350, 7], [360, 7], [370, 7], [380, 8], [400, 8]])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


model = Sequential([Dense(units=16, activation='relu', input_shape=(2,)),
                    Dense(units=8, activation='relu'),
                   Dense(units=1, activation='sigmoid')])



model.compile(optimizer='adam',
              loss='binary_crossentropy',)
model.fit(x,y , epochs=500)
x_new = np.array([[275.0,10.0]])
predict = model.predict(x_new)
prediction = model.predict(x_new)
print(f"the result :{prediction[0][0]}")

if prediction[0][0] > 0.6:
    print("it's good house ✅")
else:
    print("it's bad house ❌")