import numpy as np
from numpy import pi
import math
import tensorflow as tf
from scipy.fftpack import dct, idct, rfft, irfft
from tensorflow import keras

from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras import backend as K 

import numpy as np
import math



class DefenseFeatureDistillation():
    #Default constructor 
    def __init__(self , model, dataset, S1, S2): #to initilize, send model, dataset = 'cifar10' or 'fashion-mnist', S1=30, S2=20

        
        if dataset != 'fashion-mnist' and dataset != 'cifar-10':
            raise NameError('Dataset must be fashion-mnist or cifar-10.')

        if S1 > 100 or S2 > 100 or S1 < 0 or S2 < 0:
            raise NameError('S1 and S2 must be between 0 and 100.')

        self.num = 8
        self.q_table = np.ones((self.num,self.num))*S1 
        self.q_table[0:4,0:4] = S2
        print('Quantization Table:')
        print(self.q_table)

        self.model=model
        self.ClassNum = 10
        self.dataset=dataset

    def dct2 (block):
        return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
    def idct2(block):
        return idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')
    def rfft2 (block):
        return rfft(rfft(block.T).T)
    def irfft2(block):
        return irfft(irfft(block.T).T)


    def FD_jpeg_encode(self, input_matrix):
        output = []
        input_matrix = input_matrix*255

        n = input_matrix.shape[0]
        h = input_matrix.shape[1]
        w = input_matrix.shape[2]
        c = input_matrix.shape[3]
        horizontal_blocks_num = w / self.num
        output2=np.zeros((c,h, w))
        output3=np.zeros((n,3,h, w))    
        vertical_blocks_num = h / self.num
        n_block = np.split(input_matrix,n,axis=0)
        for i in range(1, n):
            c_block = np.split(n_block[i],c,axis =3)
            j=0
            for ch_block in c_block:
                vertical_blocks = np.split(ch_block, vertical_blocks_num,axis = 1)
                k=0
                for block_ver in vertical_blocks:
                    hor_blocks = np.split(block_ver,horizontal_blocks_num,axis = 2)
                    m=0
                    for block in hor_blocks:
                        block = np.reshape(block,(self.num,self.num))
                        block = feature_distillation.dct2(block)
                        # quantization
                        table_quantized = np.matrix.round(np.divide(block, self.q_table))
                        table_quantized = np.squeeze(np.asarray(table_quantized))
                        # de-quantization
                        table_unquantized = table_quantized*self.q_table
                        IDCT_table = feature_distillation.idct2(table_unquantized)
                        if m==0:
                            output=IDCT_table
                        else:
                            output = np.concatenate((output,IDCT_table),axis=1)
                        m=m+1
                    if k==0:
                        output1=output
                    else:
                        output1 = np.concatenate((output1,output),axis=0)
                    k=k+1
                output2[j] = output1
                j=j+1
            output3[i] = output2     

        output3 = np.transpose(output3,(0,2,1,3))
        output3 = np.transpose(output3,(0,1,3,2))
        output3 = output3/255
        output3 = np.clip(np.float32(output3),0.0,1.0)
        return output3
    

    def one_pass(self, xData):

       
        xData=xData+0.5
       
       
        if self.dataset == 'fashion-mnist':
            xData *= 255
            x = []
            for img in xData:
                resized = cv2.resize(img, (32, 32))
                x.append(resized)
            x = np.array(x) 
            x = np.expand_dims(x, axis=3)
            x = cv2.merge([x] * 3) #Create a 3-channel image by merging the grayscale image  three times
            x = np.squeeze(x, axis=3) 

            xData = x 
            xData = xData.astype('float32')
            xData /= 255

        # add one pass FD method
        xDefense = self.FD_jpeg_encode(xData)

        x = []
        if self.dataset == 'fashion-mnist': #remove extra channels from fashion-mnist

            for img in xDefense:
                resized = cv2.resize(img, (28, 28))
                x.append(resized)
    
            x = np.array(x, dtype=np.float32)
            xDefense = np.zeros( (xData.shape[0], 28, 28, 3) )
            xDefense = x[:, :, :, 0]
            xDefense=np.expand_dims(xDefense, axis=3)

        xDefense = np.array(xDefense, dtype=np.float32)
        xDefense = xDefense.astype('float32')

        xDefense = xDefense - 0.5

        return xDefense
        
    def predict(self, xData):
        
        xData = self.one_pass(xData)
    
        yDefense = self.model.predict(xData) #predict with defended outputs
        return yDefense


    def evaluate(self, xTest, yTest):
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
            if yPred[i].argmax(axis=0) != self.ClassNum and yPred[i].argmax(axis=0) != yClean[i].argmax(axis=0): #The last class is the noise class
                advAcc=advAcc+1
        advAcc=advAcc/sampleNum
        return advAcc


