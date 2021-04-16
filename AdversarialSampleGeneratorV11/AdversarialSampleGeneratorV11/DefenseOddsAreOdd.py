import numpy 
import keras
from random import seed
from random import randint
from numpy import array
from math import ceil
from math import log10
from math import sqrt
from numpy import argmax

import tensorflow 
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model
from keras.layers import Input

import sys
import os 
from tempfile import TemporaryFile

import scipy.stats
import DataManager
import tensorflow as tf
import time 

class DefenseOddsAreOdd:

    def __init__(self, modelDir, mode, mean_yz, std_yz, FPR):
        self.ModelDir = modelDir
        self.mode = mode # fashion-mnist or cifar-10
        self.mean_yz = mean_yz
        self.std_yz = std_yz
        self.FPR = FPR
        self.Model = self.build_model()
        self.t_yz = self.build_t_yz()
        self.ClassNum = 10

    def build_t_yz(self):
        # Buiding the matrix t_yz from mean_yz and std_yz
        PFR_table =dict()
        PFR_table[1] = -2.3
        PFR_table[10] = -1.28
        PFR_table[20] = -0.84
        PFR_table[30] = -0.52
        PFR_table[40] = -0.25
        PFR_table[50] = 0
        PFR_table[80] = 0.84

        t_yz = PFR_table[self.FPR]*self.std_yz + self.mean_yz

        return t_yz

    def build_model(self):
        if(self.mode == 'fashion-mnist'): #FashionMNIST
                #Loading the model
                #curr_dir = os.getcwd()
                #director = [token+"//" for token in curr_dir.split("\\")]
                #dir = ""
                #for token in director:
                #    dir += token
                #dir +="VGG16Resize=32ZeroPad=NonePertG=NoneFashionMNISTVanillia.h5"
                #dir = "C://Users//Windows//Desktop//Saved Keras Networks//Fashion MNIST//Vanillia//VGG16Resize=32ZeroPad=NonePertG=NoneFashionMNISTVanillia.h5"
                #model_vgg = DataManager.LoadModel(dir)  
                model_vgg = DataManager.LoadModel(self.ModelDir)

                #Model = keras.models.Model
                #Dense = keras.layers.Dense
                #access the logits of vgg network
                layer_name = 'batch_normalization_14'
                intermediate_layer = model_vgg.get_layer(layer_name).output
                #create logits layer
                dense_layer = keras.layers.Dense(10, name='logits', activation="linear")(intermediate_layer)

                #get the weights
                layer_name = 'z_16'
                weights = model_vgg.get_layer(layer_name).get_weights()
                #create new model with the logitis output 
                model = keras.models.Model(inputs=model_vgg.input, outputs=dense_layer)
                #set the weights 
                layer_name = 'logits'
                model.get_layer(layer_name).set_weights(weights)

                return model       
        elif(self.mode == 'cifar-10'): #CIFAR10
                #Loading the model
                #curr_dir = os.getcwd()
                #director = [token+"//" for token in curr_dir.split("\\")]
                #dir = ""
                #for token in director:
                #    dir += token
                #dir +="cifar10_ResNet6v2_model.162.h5"
                model_resnet = DataManager.LoadModel(self.ModelDir)  

                #access the logits of resnet network
                layer_name = 'flatten_1'
                intermediate_layer = model_resnet.get_layer(layer_name).output
                #create logits layer
                dense_layer = keras.layers.Dense(10, name='logits', activation="linear")(intermediate_layer)

                #get the weights
                layer_name = 'dense_1'
                weights = model_resnet.get_layer(layer_name).get_weights()
                #create new model with the logitis output 
                model = keras.models.Model(inputs=model_resnet.input, outputs=dense_layer)
                #set the weights 
                layer_name = 'logits'
                model.get_layer(layer_name).set_weights(weights)
                
                return model
        else:
            raise ValueError("Dataset not recognized.")
        
    def predict(self, xSet):
        #Output the predicted class label and decision on the groundtruth of inputs 
        sigma_noise = 0.05# the sigma for generating the noise
        n_noise = 256     
        #Compute the variables for computing in batch 
        sampleNumber = xSet.shape[0]
        batchSize = 10000 #Maximum number to run through the GPU at one time 
        totalBatchNum = int(numpy.ceil((sampleNumber *n_noise / batchSize)))
        if totalBatchNum<=0: #If number of samples less than batch size just do everything in one batch 
            print("Using single batch.")
            totalBatchNum = 1
        print("Total number of batches are:", totalBatchNum)
        print("Total number of evaluation needed:", sampleNumber *n_noise)
        singlePredY = self.Model.predict(xSet)
        finalPred = numpy.zeros((xSet.shape[0], self.ClassNum+1))
        #for x in xSet:
        for i in range(0, totalBatchNum): 
                print("Running batch #=", i)
                if i == totalBatchNum-1: #The last step is handled differently 
                    xCurrent = xSet[i*totalBatchNum:xSet.shape[0]] #Go to the very end 
                else:
                     xCurrent = xSet[i*totalBatchNum:(i+1)*totalBatchNum] #Take a step        
                #First index is sample number, second index is the jth noisy sample, third index is height, fourth index is width, fifth index is channel number
                #(B, N, H, W, C)
                xL = numpy.zeros((xCurrent.shape[0], n_noise, xCurrent.shape[1], xCurrent.shape[2], xCurrent.shape[3]), dtype= numpy.float32)
                for j in range (xCurrent.shape[0]):
                    for k in range (n_noise):
                        xL[j,k] = xCurrent[j]
                noiseL  = numpy.random.normal(0,sigma_noise,size=xL.shape)
                noisy_xL = numpy.clip(xL+noiseL, -0.5, 0.5) #Add the original samples and the noise together, clip them to -0.5, 0.5 range
                #noisy_xL=  numpy.where( (xL+noiseL < - 0.5) | (xL+noiseL>0.5),xL, xL+noiseL)
                noisy_xL = numpy.array(noisy_xL ,dtype= numpy.float32) #Convert to float32 to avoid memory errors encountered before 
                totalSampleNum = n_noise * xCurrent.shape[0] #The total number of samples to feed to the GPU is numSamples*numOfNoisyPoints
                noisy_xLSingleStack = numpy.reshape(noisy_xL,(totalSampleNum, xCurrent.shape[1], xCurrent.shape[2], xCurrent.shape[3])) #Convert from (B, N, H, W, C) to (B*N, H, W, C)
                pred_vec_noisy_x_singleStack = self.Model.predict(noisy_xLSingleStack) #Use the model to predict (will use GPU)
                #Will reshape from (B*N, H, W, C) => (B, N, H, W, C)
                pred_vec_noisy_x = numpy.reshape(pred_vec_noisy_x_singleStack, (xCurrent.shape[0], n_noise, self.ClassNum))
                tempIndex = i*totalBatchNum
                for j in range (xCurrent.shape[0]):
                    pred_vec_x = numpy.array([singlePredY[tempIndex]]*n_noise)
                    result =  self.SingleSampleCompute(pred_vec_x, pred_vec_noisy_x[j])
                    if result[0] == True:
                        finalPred[tempIndex, self.ClassNum] = 1.0 #Adversarial sample 
                    else:
                        finalPred[tempIndex, int(result[1])] = 1.0 #Clean sample 
                    tempIndex = tempIndex + 1  
        return finalPred

    def SingleSampleCompute(self, pred_vec_x, pred_vec_noisy_x):
        y = numpy.argmax(pred_vec_x[0]) # get the predicted label y from the plain model
        gbar_yz = numpy.zeros(self.ClassNum) 

        #compute the gbar_yz for all z 
        for z in range(self.ClassNum):
                g_yz_xnoise =  (pred_vec_noisy_x[:,z] - pred_vec_noisy_x[:,y]) - (pred_vec_x[:,z]-pred_vec_x[:,y])
                #normalize g_yz_xnoise 
                gbar_yz_xnoise = (g_yz_xnoise - self.mean_yz[y][z])/self.std_yz[y][z]
                gbar_yz[z]  = numpy.mean(gbar_yz_xnoise)

        #verifying the predicted class label y from the plain model and correct it if it is malicious
        flag = False # not an adversarial example

        gbar_yz_v = - 10000 
        z_store  = -1 
        #checking the input -- equation (5) in the paper
        for z in range(self.ClassNum):
            if(z!=y):
                gap = gbar_yz[z] - self.t_yz[y][z]
                if(gap>gbar_yz_v):
                    gbar_yz_v = gap 
                    z_store = z 
        #make decision on the input x and suggest the class label if the input is malicious
        if(gbar_yz_v>=0):
            flag = True # x is an adversarial
            #print("x is an adversarial examples")
            #print("Suggested class label:", z_store)
            return (flag,z_store)

        if(flag==False):
            ##print("x is a benign sample")
            ##print("Suggested class label: ", y)
            return (flag,y)

    def evaluate(self, xTest, yTest):
        accuracy=0
        sampleSize=xTest.shape[0]
        multiModelOutput=self.predict(xTest)
        for i in range(0, sampleSize):
            if(multiModelOutput[i].argmax(axis=0)==yTest[i].argmax(axis=0)):
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
            if yPred[i].argmax(axis=0) != self.ClassNum and yPred[i].argmax(axis=0) != yClean[i].argmax(axis=0): #The last class is the noise class
                advAcc=advAcc+1
        advAcc=advAcc/sampleNum
        return advAcc   