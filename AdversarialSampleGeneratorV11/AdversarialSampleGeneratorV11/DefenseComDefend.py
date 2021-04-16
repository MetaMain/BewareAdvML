import tensorflow as tf
from tensorflow import keras
import Canton as ct
from Canton import *
import numpy as np
import math
from keras.models import load_model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from keras import backend as K 

class DefenseComDefend():
    #Default constructor 
    def __init__(self , model, dirCom, dirRec, fashion):
        print("Warning: This defense is only configured to work with CIFAR-10 or Fashion-MNIST currently.")
        self.model=model
        self.fashion = fashion #flag for fashion-mnist dataset
        self.ClassNum = 10
        self.dirCom = dirCom #encoder weights directory
        self.dirRec = dirRec #decoder weights directory
        self.com = self.ComCNN()
        self.res = self.ResCNN() #com and res classes

    def ComCNN(self): 
            c=Can()
            def conv(nip,nop,flag=True):
                c.add(Conv2D(nip,nop,k=3,usebias=True))
                if flag:
                    # c.add(BatchNorm(nop))
                    c.add(Act('elu'))
            if self.fashion == 1:
                c.add(Lambda(lambda x:x-0.5))
                conv(1,32)
                conv(32,64)
                conv(64,128)
                conv(128,256)
                conv(256, 128)
                conv(128,64)
                conv(64, 64)
                conv(64,32)
                conv(32, 4,flag=False)
                c.chain()
            elif self.fashion == 0:
                 c.add(Lambda(lambda x:x-0.5))
                 conv(3,16)
                 conv(16,32)
                 conv(32,64)
                 conv(64,128)
                 conv(128,256)
                 conv(256,128)
                 conv(128,64)
                 conv(64,32)
                 conv(32,12,flag=False)
                 c.chain()
            return c

    def ResCNN(self):
            c=Can()
            def conv(nip,nop,flag=True):
                c.add(Conv2D(nip,nop,k=3,usebias=True))
                if flag:
                    # c.add(BatchNorm(nop))
                    c.add(Act('elu'))
            if self.fashion == 1:
                conv(4,32)
                conv(32,64)
                conv(64,128)
                conv(128, 256)
                conv(256, 128)
                conv(128,64)
                conv(64,64)
                conv(64,32)
                conv(32, 1, flag=False)
                c.add(Act('sigmoid'))
                c.chain()
            elif self.fashion == 0 :
                conv(12,32)
                conv(32,64)
                conv(64,128)
                conv(128,256)
                conv(256,128)
                conv(128,64)
                conv(64,32)
                conv(32,16)
                conv(16,3,flag=False)
                c.add(Act('sigmoid'))
                c.chain()
            return c

    def get_defense(self, com, res): 
        if self.fashion == 1:
            x = ph([None,None,1])
        elif self.fashion == 0:
            x = ph([None,None,3])
        x = tf.clip_by_value(x,clip_value_max=1.,clip_value_min=0.)
        code_noise = tf.Variable(1.0)
        linear_code = com(x)
        # add gaussian before sigmoid to encourage binary code
        noisy_code = linear_code - \
            tf.random_normal(stddev=code_noise,shape=tf.shape(linear_code))
        binary_code = Act('sigmoid')(noisy_code)
        y = res(binary_code)
        set_training_state(False)
        quantization_threshold = tf.Variable(0.5)
        binary_code_test = tf.cast(binary_code>quantization_threshold,tf.float32)
        y_test = res(binary_code_test)

        def test(batch,quanth):
            sess = ct.get_session()
            res = sess.run([binary_code_test,y_test,binary_code,y,x],feed_dict={
                x:batch,
                quantization_threshold:quanth,
            })
            return res
        return test

    def change(self, x):
        x *=255
        x=x.astype('uint8')
        img=x[0]
        return img

    def predict(self, xData):   
        #Some extra error handling just in case a list was passed in 
        #if isinstance(xData, list):
        #    xDataNP = np.asarray(xData)
        #    xData = xDataNP
        xData = xData + 0.5     
        test = self.get_defense(self.com,self.res)
        get_session().run(ct.gvi())
        self.com.load_weights(self.dirCom)
        self.res.load_weights(self.dirRec)
        #print("Compressing & reconstructing the input:")
        x_defense = []
        ctr = 0
        threshold = 0.5
        for im in xData:     
            minibatch =[im]
            minibatch=np.array(minibatch)
            code, rec, code2, rec2, x= test(minibatch,threshold)
            if self.fashion == 1:
                img = self.change(rec2)
            elif self.fashion == 0:
                img = self.change(rec)
            out = img
            x_defense.append(out)
            ctr = ctr + 1
            #if (ctr % 1000 == 0):
            #    print(ctr, "/", xData.shape[0])
        x_defense = np.asarray(x_defense)
        if self.fashion == 1: 
            x_defense = np.squeeze(x_defense, axis=3)
        x_defense = x_defense.astype('float32')
        x_defense /= 255
        x_defense=x_defense-0.5 #Do a subtraction of 0.5 to match Carlini and others data format 
        if self.fashion == 1:
            x_defense_final = np.expand_dims(x_defense, axis=3) #make array same dimensions as CIFAR10
        elif self.fashion == 0:
            x_defense_final = x_defense
        prediction = self.model.predict(x_defense_final) #predict with defended outputs                                      
        return prediction

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


