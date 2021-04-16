import DataManager
import VggNetworkConstructor 
import ResNetConstructor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy as np


class DefenseADPNetwork():
  #Default constructor 
    def __init__(self , dirModel, dataset, num_models, lamda, log_det_lamda):

        self.dirModel = dirModel
        self.dataset = dataset 
        self.num_models = num_models
        self.lamda = lamda
        self.log_det_lamda = log_det_lamda

        self.num_classes = 10
        self.log_offset = 1e-20
        self.det_offset = 1e-6
        self.LoadModel()

    def predict(self, xData):
        prediction = self.model_ensemble.predict(xData)
        return prediction


    def LoadModel(self):       
        model_out = []
        if self.dataset == 'fashion-mnist': #using tensorflow functions for the model
            input_shape = (28, 28, 1)
            img_rows=input_shape[0]
            img_cols=input_shape[1]
            colorChannelNum=input_shape[2]
            input_img = tf.keras.layers.Input(shape=(img_rows, img_cols, colorChannelNum))
            resizeValue=32
            zeroPadValue=None
            bMatrix=None
            pertValue=None
            aMatrixForKerasLayer =None 
            for i in range(self.num_models):
                out = VggNetworkConstructor.GenerateBasePrivateVgg16Model(input_shape, input_img, self.num_classes, resizeValue, zeroPadValue, aMatrixForKerasLayer, bMatrix, pertValue)
                model_out.append(out)
            model_output = tf.keras.layers.concatenate(model_out) 
            model = tf.keras.models.Model(input_img, model_output)
            model_ensemble = tf.keras.layers.Average()(model_out)
            model_ensemble = tf.keras.models.Model(inputs=input_img, outputs=model_ensemble)        
        elif self.dataset == 'cifar-10': #using keras functions for the model because the model constructor uses keras and it is not compatible with tensorflow
            input_shape = (32,32,3)
            #input_shape = xData[1].shape
            print(input_shape)
            model_input = Input(shape=input_shape)
            print(model_input)
            model_dic = {}
            model_out = []
            complex = 6
            for i in range(self.num_models):
                model_dic[str(i)] = model_dic[str(i)] = ResNetConstructor.resnet_v2(input=model_input, complexityParameter = complex, num_classes=self.num_classes, dataset=self.dataset)
                model_out.append(model_dic[str(i)][2])
            model_output = tf.keras.layers.concatenate(model_out) 
            model = tf.keras.models.Model(model_input, model_output)
            model_ensemble = tf.keras.layers.Average()(model_out)
            model_ensemble = tf.keras.models.Model(inputs=model_input, outputs=model_ensemble)    
        else:
            raise ValueError("Dataset not recognized.")
        model.load_weights(self.dirModel)
        self.model_ensemble = model_ensemble

    def evaluate(self, xTest, yTest):
        print(yTest.shape)
        accuracy=0
        sampleSize=xTest.shape[0]
        predictOutput=self.predict(xTest)
        for i in range(0, sampleSize):
            if(predictOutput[i].argmax(axis=0)==yTest[i].argmax(axis=0)):
                accuracy=accuracy+1
        accuracy=accuracy/sampleSize
        return accuracy

    #the network is fooled if we don't have a noise class label AND it gets the wrong label 
    #Returns attack success rate 
    def evaluateAdversarialAttackSuccessRate(self, xAdv, yClean):
        sampleNum=xAdv.shape[0]
        yPred=self.predict(xAdv)
        advAcc=0
        for i in range(0, sampleNum):
            #The attack wins only if we don't correctly label the sample AND the sample isn't given the nosie class label
            if yPred[i].argmax(axis=0) != self.num_classes and yPred[i].argmax(axis=0) != yClean[i].argmax(axis=0): #The last class is the noise class
                advAcc=advAcc+1
        advAcc=advAcc/sampleNum
        return advAcc