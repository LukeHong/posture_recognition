from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.optimizers import SGD, Adam
import numpy as np
import datetime

#train data
x_train = []
y_train = []
train_file = open('data/posture_train.txt', 'r')
angle=[]
target=[]
for line in train_file.readlines():
    row = line.split()
    angle.append(np.array([float(x) for x in row[:4]]))
    target.append(np.array([float(y) for y in row[4:]]))
train_file.close()
x_train = np.array(angle)
y_train = np.array(target)

#test data
x_test = []
y_test = []
test_file = open('data/posture_test.txt', 'r')
angle=[]
target=[]
for line in test_file.readlines():
    row = line.split()
    angle.append(np.array([float(x) for x in row[:4]]))
    target.append(np.array([float(y) for y in row[4:]]))
test_file.close()
x_test = np.array(angle)
y_test = np.array(target)

#model
model = Sequential()
model.add(Dense(input_dim=4, output_dim=1000))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=1000))
model.add(Activation('sigmoid'))
model.add(Dense(output_dim=9))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=200, nb_epoch=10)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', round(score[1], 4))

model_result = model.to_json()
time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
accuracy = str(int(score[1]*10000))
filename = 'model_{0}_acc{1}.json'.format(time, accuracy)
with open('result/'+filename, 'w') as file:
    file.write(model_result)
    file.close()
