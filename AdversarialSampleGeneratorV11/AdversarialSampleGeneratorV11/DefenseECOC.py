import tensorflow as tf
#from cleverhans.utils_keras import KerasModelWrapper as CleverHansKerasModelWrapper
#from tensorflow.keras.layers import Layer, BatchNormalization, Dropout, Lambda, Input, Dense, Conv2D, Flatten, Activation, Concatenate, concatenate, GaussianNoise
#from tensorflow.keras.utils import plot_model
#from tensorflow.keras import regularizers
import tensorflow as tf
import scipy.linalg
import numpy as np
from math import ceil
import os
import ECOCModel as Model


#tf.set_random_seed(1) 

import DataManager

#COMMENTS FOR TANH ENSEMBLE MODEL: 
#1. num_chunks refers to how many models comprise the ensemble (4 is used in the paper); code_length/num_chunks shoould be an integer
#2. output_activation is the function to apply to the logits
#   a. one can use anything which gives support to positive and negative values (since output code has +1/-1 elements); tanh or identity maps both work
#   b. in order to alleviate potential concerns of gradient masking with tanh, one can use identity as well
#3. M is the actual coding matrix (referred to in the paper as H).  Each row is a codeword
#   note that any random shuffle of a Hadmard matrix's rows or columns is still orthogonal
#4. There is nothing particularly special about the seed (which effectively determines the coding matrix). 
#   We tried several seeds from 0-60 and found that all give comparable model performance (e.g. benign and adversarial accuracy). 




class DefenseECOC():
    #Default constructor 
    def __init__(self, dataset, Session, model_path, BATCH_NORMALIZATION_FLAG, DATA_AUGMENTATION_FLAG, batch_size, num_chunks): 
        self.Session = Session
        self.ClassNum = 10
        if dataset == 'fashion-mnist':
            ########DATASET-SPECIFIC PARAMETERS: F-MNIST
            num_channels = 1
            inp_shape = (28,28,1)
            num_classes=10
            lr=3e-4
            ##MODEL DEFINITION PARAMETERS
            num_filters_std = [64, 64, 64]
            num_filters_ens=[32, 32, 32]
            num_filters_ens_2=4
            dropout_rate_std=0.0; dropout_rate_ens=0.0; weight_decay = 0 
            noise_stddev = 0.3; blend_factor=0.3; 
            model_rep_ens=2

        elif dataset == 'cifar10':
            ##########DATASET-SPECIFIC PARAMETERS: CIFAR10
            num_channels = 3
            inp_shape = (32,32,3)
            num_classes=10
            lr=2e-4
            #MODEL DEFINITION PARAMETERS
            num_filters_std = [32, 64, 128]
            num_filters_ens=[32, 64, 128]
            num_filters_ens_2=16
            dropout_rate_std=0.0
            dropout_rate_ens=0.0
            weight_decay = 0 
            noise_stddev = 0.032
            blend_factor=0.032
            model_rep_ens=2
        else:
        
            raise NameError('Dataset must be fashion-mnist or cifar10.')
        
        #data_dict = {'X_train':X_train, 'Y_train_cat':Y_train, 'X_test':X_test, 'Y_test_cat':Y_test}

        name = 'tanh_32_diverse'+'_'+dataset; seed = 59; code_length=32; num_codes=code_length; num_chunks; base_model=None;

        def output_activation(x):
            return tf.nn.tanh(x)

        M = scipy.linalg.hadamard(code_length).astype(np.float32) #coding matrix 
        M[np.arange(0, num_codes,2), 0]= -1#replace first col, which for scipy's Hadamard construction is always 1, hence not a useful classifier; this change still ensures all codewords have dot product <=0; since our decoder ignores negative correlations anyway, this has no net effect on probability estimation
        np.random.seed(seed)
        np.random.shuffle(M)
        idx=np.random.permutation(code_length)
        M = M[0:num_codes, idx[0:code_length]]
        
        
        params_dict = {'BATCH_NORMALIZATION_FLAG':BATCH_NORMALIZATION_FLAG, 'DATA_AUGMENTATION_FLAG':DATA_AUGMENTATION_FLAG, 'M':M, 'base_model':base_model, 'num_chunks':num_chunks, 'model_rep': model_rep_ens, 'output_activation':output_activation, 'num_filters_ens':num_filters_ens, 'num_filters_ens_2':num_filters_ens_2,'batch_size':batch_size,  'blend_factor':blend_factor, 'inp_shape':inp_shape, 'noise_stddev':noise_stddev, 'name':name}

        self.model = Model.Model(params_dict)
        self.model.loadFullModel(model_path)
 
    
    def predict(self, xData):
        totalCount = 0
        sampleSize=xData.shape[0]
        probs_benign_list=[]  
        for rep in np.arange(0, sampleSize, 1000):
                x = xData[rep:rep+1000]
                probs_benign = self.predictECC(x)
                probs_benign_list += list(np.argmax(probs_benign[:,0:self.ClassNum], 1))
                #probs_benign_list += list(np.argmax(probs_benign, 1))
        predictOutput = np.asarray(probs_benign_list)
        yPred = np.zeros((sampleSize, self.ClassNum))
        for i in range(0, sampleSize):
            yPred[i, int(predictOutput[i])] = 1.0
        return yPred


    #This is the predict method that gives the full code (not class label) back 
    def predictECC(self, xData):
        sess = self.Session
        yDefense = sess.run(self.model.model_full(tf.convert_to_tensor(xData)))
        return yDefense

    def evaluate(self, xTest, yTest):
        predictOutput = self.predict(xTest)
        sampleSize=xTest.shape[0]
  
        #probs_benign_list=[]
        
        #for rep in np.arange(0, sampleSize, 1000):
        #        x = xTest[rep:rep+1000]
        #        probs_benign = self.predictECC(x)
        #        probs_benign_list += list(np.argmax(probs_benign, 1))
        #predictOutput = np.asarray(probs_benign_list)

        accuracy = 0
        for i in range(0, sampleSize):
            if(predictOutput[i].argmax(axis=0)==yTest[i].argmax(axis=0)):
                accuracy=accuracy+1

         
        accuracy=accuracy/sampleSize

        return accuracy

    #the network is fooled if we don't have a noise class label AND it gets the wrong label 
    #Returns attack success rate 
    def evaluateAdversarialAttackSuccessRate(self, xAdv, yClean):

        sampleNum=xAdv.shape[0]
        predictOutput = self.predict(xAdv)
        #probs_benign_list=[]
        
        #for rep in np.arange(0, sampleNum, 1000):
        #        x = xAdv[rep:rep+1000]
        #        probs_benign = self.predictECC(x)
        #        probs_benign_list += list(np.argmax(probs_benign, 1))
        #predictOutput = np.asarray(probs_benign_list)

        advAcc = 0
        for i in range(0, sampleNum):
            if predictOutput[i].argmax(axis=0) != self.ClassNum and predictOutput[i].argmax(axis=0) != yClean[i].argmax(axis=0): #The last class is the noise class
                advAcc=advAcc+1

        advAcc=advAcc/sampleNum
        return advAcc


