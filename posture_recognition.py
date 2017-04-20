import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam

#load file into pd.DataFrame
with open('data/posture_train.txt', 'r') as train_file:
    train_df = pd.read_csv(train_file, header=None, delim_whitespace=True)

with open('data/posture_test.txt', 'r') as test_file:
    test_df = pd.read_csv(test_file, header=None, delim_whitespace=True)


#dataset

#train data
x = []
y = []
for index, row in train_df.iterrows():
    x.append(np.array(row[:4]))
    y.append(np.array(row[4:]))
x_train = np.array(x)
y_train = np.array(y)

#test data
x = []
y = []
for index, row in test_df.iterrows():
    x.append(np.array(row[:4]))
    y.append(np.array(row[4:]))
x_test = np.array(x)
y_test = np.array(y)

layer = 5
units = 16
activation = 'relu'

#model
model = Sequential()
model.add(Dense(input_dim=4, units=units))
model.add(Activation(activation))
for i in range(layer):
    model.add(Dense(units=units))
    model.add(Activation(activation))
model.add(Dense(units=9))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

#train
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=20,
                    shuffle=True,
                    validation_split=0.1)


#result
score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', round(score[1], 4)*100)


#save model
model_result = model.to_json()
time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
accuracy = str(int(score[1]*10000))
filename = 'model_{0}_acc{1}'.format(time, accuracy)
with open('result/'+filename+'.json', 'w') as file:
    file.write(model_result)
model.save_weights('result/'+filename+'.h5')

# plot result
loss = history.history.get('loss')
acc = history.history.get('acc')

plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss)), loss)
plt.title('Loss')
#plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc)
plt.title('Accuracy')
plt.show()
#plt.savefig('{}_{}*{}.jpg'.format(activation, layer, units),dpi=300,format='jpg')
plt.close()
