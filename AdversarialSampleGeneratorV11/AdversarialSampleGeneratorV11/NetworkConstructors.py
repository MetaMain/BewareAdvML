#This contains the model constructors for the black box attacks 
import tensorflow as tf
from tensorflow import keras
Model = keras.models.Model
Sequential=keras.models.Sequential
Dense = keras.layers.Dense
Dropout=keras.layers.Dropout
Activation = keras.layers.Activation
Flatten = keras.layers.Flatten
BatchNormalization= keras.layers.BatchNormalization
Conv2D = keras.layers.Conv2D
Dropout=keras.layers.Dropout
MaxPooling2D=keras.layers.MaxPooling2D
Input=keras.layers.Input
regularizers=keras.regularizers
from tensorflow.python.keras import optimizers

#This is the CIFAR10 network used in Carlini's code: https://github.com/carlini/nn_robust_attacks
def ConstructCarliniNetwork(inputShape, numClasses):
    params=[64, 64, 128, 128, 256, 256]
    model = Sequential()    
    model.add(Conv2D(params[0], (3, 3), input_shape=inputShape))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(numClasses, name="dense_2"))
    model.add(Activation('softmax'))
    learningRate = 0.0001
    opt=tf.train.AdamOptimizer(learning_rate=learningRate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

#This is MNIST network A from Papernot's blackbox attack  
def PapernotMNISTNetworkAConstructor(input_shape, num_classes):
    input_img = Input(shape=input_shape)
    conv1=Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=input_shape)(input_img)
    max1=MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2=Conv2D(64, kernel_size=(2, 2), activation='relu')(max1)
    max2=MaxPooling2D(pool_size=(2, 2))(conv2)
    flat=Flatten()(max2)
    dense1=Dense(200, activation='relu')(flat)
    dense2=Dense(200, activation='relu')(dense1)
    dense3=Dense(num_classes, activation='softmax')(dense2)
    model = Model(inputs=input_img, outputs=dense3)
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy']) 
    return model 



