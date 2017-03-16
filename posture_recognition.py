import datetime

import numpy as np
import pandas as pd

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

layer = 8

#model
model = Sequential()
model.add(Dense(input_dim=4, output_dim=16))
model.add(Activation('softplus'))
for i in range(layer):
    model.add(Dense(16))
    model.add(Activation('softplus'))
model.add(Dense(output_dim=9))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])


#train
model.fit(x_train, y_train, batch_size=64, nb_epoch=40)


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
