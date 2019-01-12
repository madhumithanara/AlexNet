import tensorflow as tf
from tensorflow import keras

class AlexNet:
    def __init__(self):
        self.model=keras.Sequential()
        self.model.add(keras.layers.Conv2D(96, (11,11),strides=(4,4), activation='relu',input_shape = (227,227,3),name='block1_conv1'))
        self.model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='block1_pool'))
        self.model.add(keras.layers.BatchNormalization())
        
        self.model.add(keras.layers.Conv2D(256,(5,5),padding='same',activation='relu'))
        self.model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='block2_pool'))
        self.model.add(keras.layers.Conv2D(384,(3,3),padding='same',activation='relu'))
        self.model.add(keras.layers.Conv2D(384,(3,3),padding='same',activation='relu'))
        self.model.add(keras.layers.Conv2D(256,(3,3),padding='same',activation='relu'))
        self.model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name='block3_pool'))
        self.model.add(tf.layers.Flatten())

        self.model.add(keras.layers.Dense(4096,activation='relu'))
        self.model.add(keras.layers.Dense(4096,activation='relu'))
        self.model.add(keras.layers.Dense(1000,activation='softmax'))
        self.model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model1=AlexNet()
model1.model.summary()
