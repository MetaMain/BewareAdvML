#This class conatins methods for loading the clean data, saved adversarial data and saved models 
import numpy
import tensorflow
from tensorflow import keras
from keras.datasets import cifar10, cifar100
import pickle
from tensorflow.keras import backend as K 

def LoadCleanDataMNIST():
    (xTrain, yTrain), (xTest, yTest) = tensorflow.keras.datasets.mnist.load_data()
    xTrain = xTrain.astype('float32')
    xTest = xTest.astype('float32')
    xTrain /= 255 #Convert the pixel values in the training data between 0 and 1 range 
    xTest /= 255 #Conver the pixel values in the testing data between 0 and 1 range 
    xTrain=xTrain-0.5 #Do a subtraction of 0.5 to match Carlini and others data format 
    xTest=xTest-0.5 #Do a subtraction of 0.5 to match Carlini and others data format 
    xTrainFinal=numpy.expand_dims(xTrain, axis=3) #make array same dimensions as CIFAR10
    xTestFinal=numpy.expand_dims(xTest, axis=3) #make array same dimensions as CIFAR10
    yTrain = keras.utils.to_categorical(yTrain, 10) # convert class vectors to binary class matrices
    yTest = keras.utils.to_categorical(yTest, 10)
    return xTrainFinal, yTrain, xTestFinal, yTest

#Load clean (non-adversarial) CIFAR10 data using Keras
def LoadCleanDataCIFAR10():
    (xTrain, yTrain), (xTest, yTest) = cifar10.load_data() # the data split between train and test sets
    xTrain = xTrain.astype('float32')
    xTest = xTest.astype('float32')
    xTrain /= 255 #Convert the pixel values in the training data between 0 and 1 range 
    xTest /= 255 #Conver the pixel values in the testing data between 0 and 1 range 
    xTrain=xTrain-0.5 #Do a subtraction of 0.5 to match Carlini and others data format 
    xTest=xTest-0.5 #Do a subtraction of 0.5 to match Carlini and others data format 
    yTrain = keras.utils.to_categorical(yTrain, 10) # convert class vectors to binary class matrices
    yTest = keras.utils.to_categorical(yTest, 10)
    return xTrain, yTrain, xTest, yTest

#Load FashionMNIST
def LoadCleanDataFashionMNIST():
    (xTrain, yTrain), (xTest, yTest) = tensorflow.keras.datasets.fashion_mnist.load_data()
    xTrain = xTrain.astype('float32')
    xTest = xTest.astype('float32')
    xTrain /= 255 #Convert the pixel values in the training data between 0 and 1 range 
    xTest /= 255 #Conver the pixel values in the testing data between 0 and 1 range 
    xTrain=xTrain-0.5 #Do a subtraction of 0.5 to match Carlini and others data format 
    xTest=xTest-0.5 #Do a subtraction of 0.5 to match Carlini and others data format 
    xTrainFinal=numpy.expand_dims(xTrain, axis=3) #make array same dimensions as CIFAR10
    xTestFinal=numpy.expand_dims(xTest, axis=3) #make array same dimensions as CIFAR10
    yTrain = keras.utils.to_categorical(yTrain, 10) # convert class vectors to binary class matrices
    yTest = keras.utils.to_categorical(yTest, 10)
    return xTrainFinal, yTrain, xTestFinal, yTest

def LoadCleanDataCIFAR100(): 
    (xTrain, yTrain), (xTest, yTest) = cifar100.load_data() # the data split between train and test sets
    xTrain = xTrain.astype('float32')
    xTest = xTest.astype('float32')
    xTrain /= 255 #Convert the pixel values in the training data between 0 and 1 range 
    xTest /= 255 #Conver the pixel values in the testing data between 0 and 1 range 
    xTrain=xTrain-0.5 #Do a subtraction of 0.5 to match Carlini and others data format 
    xTest=xTest-0.5 #Do a subtraction of 0.5 to match Carlini and others data format 
    yTrain = keras.utils.to_categorical(yTrain, 100) # convert class vectors to binary class matrices
    yTest = keras.utils.to_categorical(yTest, 100)
    return xTrain, yTrain, xTest, yTest

#Saves xClean, yClean, xAdv, yAdv 
def SaveAdversarialData(attackName, xClean, yClean, xAdv, yAdv=None):
    xAdvFile = open('xAdv'+attackName+'.pkl', 'wb')
    pickle.dump(xAdv, xAdvFile)
    xAdvFile.close()
    xCleanFile = open('xClean'+attackName+'.pkl', 'wb')
    pickle.dump(xClean, xCleanFile)
    xCleanFile.close()
    yCleanFile = open('yClean'+attackName+'.pkl', 'wb')
    pickle.dump(yClean, yCleanFile)
    yCleanFile.close()
    if yAdv is None:
        k = 5 #do nothing 
    else:
        yAdvFile = open('yAdv'+attackName+'.pkl', 'wb') 
        pickle.dump(yAdv, yAdvFile)
        yAdvFile.close()

#Load adversarial data
def LoadAdversarialData(advFileDir, attackName, targeted=False):
    xClean=numpy.load(advFileDir+"//xClean"+attackName+".pkl") #Original images
    yClean=numpy.load(advFileDir+"//yClean"+attackName+".pkl") #Original (ground truth) labels
    xAdv=numpy.load(advFileDir+"//xAdv"+attackName+".pkl") #Adversarial images 
    if targeted==True:
        yAdv=numpy.load(advFileDir+"//yAdv"+attackName+".pkl") #Adversarial labels 
        return xClean, yClean, xAdv, yAdv
    else:
        return xClean, yClean, xAdv

#Loads a pre-trained Keras model, here optimizer is assumed to be ADAM with cross-entropy
def LoadModel(modelFileDir):
    K.set_learning_phase(0) #This fixes problems with drop out and batchnorm for Cleverhans with TF 1.X
    model=tensorflow.keras.models.load_model(modelFileDir,  custom_objects={"tensorflow": tensorflow})
    learningRate = 0.0001
    opt=tensorflow.train.AdamOptimizer(learning_rate=learningRate)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

#This loads the robust or non-robust data, note the test data is still from original CIFAR10
def LoadCIFAR10Robust(dir, robustName):
    #First get test data 
    (xTrain2, yTrain2), (xTest, yTest) = cifar10.load_data() # the data split between train and test sets
    xTest = xTest.astype('float32')
    xTest /= 255 #Conver the pixel values in the testing data between 0 and 1 range 
    xTest=xTest-0.5 #Do a subtraction of 0.5 to match Carlini and others data format 
    yTest = keras.utils.to_categorical(yTest, 10)

    #Load the MIT data 
    np_load_old = numpy.load
    numpy.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k) #Allows the data to be loaded 
    xTrain = numpy.load(dir+"//xTrain"+robustName+'.pkl')
    xTrain = xTrain.astype('float32')
    xTrain = xTrain-0.5
    yTrain = numpy.load(dir+"//yTrain"+robustName+'.pkl')
    yTrain = keras.utils.to_categorical(yTrain, 10) # convert class vectors to binary class matrices
    numpy.load = np_load_old #Restore the numpy configuration 
    return xTrain, yTrain, xTest, yTest



