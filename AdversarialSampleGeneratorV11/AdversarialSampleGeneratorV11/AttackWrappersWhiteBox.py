import tensorflow 
from tensorflow import keras
import numpy
import time
import DataManager
from l2_attack import CarliniL2
from li_attack import CarliniLi
from l0_attack import CarliniL0
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.utils_tf import jacobian_graph, jacobian_augmentation
from cleverhans.loss import CrossEntropy
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import MomentumIterativeMethod
from cleverhans.attacks import ElasticNetMethod 

#This is a slightly modified version of Carlini's CIFARModel class that ONLY takes a VGG16 network and 
#converts it to a form that can work with the Carlini attacks by removing the softmax layer
class CIFARModel:
    #Do not initalized with anything other than VGG16 as the removal of the softmax layer is hard coded 
    def __init__(self, publicModel=None):
        self.num_channels = publicModel.layers[0].input_shape[3]
        self.image_size = publicModel.layers[0].input_shape[1]
        self.num_labels = 10
        #Just take up to (but not including) the softmax 
        Model = keras.models.Model
        Dense = keras.layers.Dense
        layer_name = 'dense_2' #This part is hard coded and my throw an error 
        intermediate_layer_model = Model(inputs=publicModel.input, outputs=publicModel.get_layer(layer_name).output)
        learningRate = 0.0001
        opt=tensorflow.train.AdamOptimizer(learning_rate=learningRate)
        intermediate_layer_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.model=intermediate_layer_model
    def predict(self, data):
        return self.model(data)

#Targeted Carlini L2 attack on Keras model using code from: https://github.com/carlini/nn_robust_attacks
def CarliniL2Targeted(session, attackBatchSize, maxIterations, publicModel, xClean, yAdv, boxMin, boxMax):
    #Load the model using Carlini's model class
    carliniModel=CIFARModel(publicModel) #This also chops off the softmax layer for VGG16 (hardcoded)
    attack = CarliniL2(session, carliniModel, batch_size=attackBatchSize, max_iterations=maxIterations, confidence=0, boxmin = boxMin, boxmax = boxMax) #Default is targeted=true for this method
    timestart = time.time()
    xAdv = attack.attack(xClean, yAdv)
    timeend = time.time()
    print("Carlini L2 Targeted took ",timeend-timestart," seconds to run ", len(xAdv), " samples.")
    return xAdv

#Untargeted Carlini L2 attack on Keras model using code from: https://github.com/carlini/nn_robust_attacks
def CarliniL2NOTarget(session, attackBatchSize, maxIterations, publicModel, xClean, yClean, boxMin, boxMax):
    #Load the model using Carlini's model class
    carliniModel=CIFARModel(publicModel) #This also chops off the softmax layer for VGG16 (hardcoded)
    attack = CarliniL2(session, carliniModel, batch_size=attackBatchSize, max_iterations=maxIterations, confidence=0, targeted=False, boxmin = boxMin, boxmax = boxMax) #Default is targeted=true for this method
    timestart = time.time()
    xAdv = attack.attack(xClean, yClean) #Here input should be the clean data and clean class labels 
    timeend = time.time()
    print("Carlini L2 Untargeted took ",timeend-timestart," seconds to run ", len(xAdv), " samples.")
    return xAdv

#Targeted Carlini LI attack on Keras model using code from: https://github.com/carlini/nn_robust_attacks
def CarliniLITargeted(session, maxIterations, publicModel, xClean, yAdv):
    #Load the model using Carlini's model class
    carliniModel=CIFARModel(publicModel) #This also chops off the softmax layer for VGG16 (hardcoded)
    attack = CarliniLi(session, carliniModel, max_iterations=maxIterations)
    timestart = time.time()
    xAdv = attack.attack(xClean, yAdv)
    timeend = time.time()
    print("Carlini LI Targeted took ",timeend-timestart," seconds to run ", len(xAdv), " samples.")
    return xAdv

#Untargeted Carlini LI attack on Keras model using code from: https://github.com/carlini/nn_robust_attacks
def CarliniLINOTarget(session, maxIterations, publicModel, xClean, yClean):
    #Load the model using Carlini's model class
    carliniModel=CIFARModel(publicModel) #This also chops off the softmax layer for VGG16 (hardcoded)
    attack = CarliniLi(session, carliniModel, targeted=False, max_iterations=maxIterations)
    timestart = time.time()
    xAdv = attack.attack(xClean, yClean)
    timeend = time.time()
    print("Carlini LI Untargeted took ",timeend-timestart," seconds to run ", len(xAdv), " samples.")
    return xAdv

#Targeted Carlini L0 attack on Keras model using code from: https://github.com/carlini/nn_robust_attacks
def CarliniL0Targeted(session, maxIterations, publicModel, xClean, yAdv):
    #Load the model using Carlini's model class
    carliniModel=CIFARModel(publicModel) #This also chops off the softmax layer for VGG16 (hardcoded)
    attack = CarliniL0(session, carliniModel, max_iterations=maxIterations)
    timestart = time.time()
    xAdv = attack.attack(xClean, yAdv)
    timeend = time.time()
    print("Carlini L0 Targeted took ",timeend-timestart," seconds to run ", len(xAdv), " samples.")
    return xAdv

#Untargeted Carlini L0 attack on Keras model using code from: https://github.com/carlini/nn_robust_attacks
def CarliniL0NOTarget(session, maxIterations, publicModel, xClean, yClean):
    #Load the model using Carlini's model class
    carliniModel=CIFARModel(publicModel) #This also chops off the softmax layer for VGG16 (hardcoded)
    attack = CarliniL0(session, carliniModel, targeted=False, max_iterations=maxIterations)
    timestart = time.time()
    xAdv = attack.attack(xClean, yClean)
    timeend = time.time()
    print("Carlini L0 Untargeted took ",timeend-timestart," seconds to run ", len(xAdv), " samples.")
    return xAdv

#Non targeted Iterative Fast Gradient Sign Method (IFGSM) using FGSM code from Cleverhans: https://github.com/tensorflow/cleverhans
def IFGSMNOTarget(session, maxIterations, publicModel,  xClean, yClean, eps, clipMin, clipMax):
    imgRows=xClean.shape[1]
    imgCols=xClean.shape[2]
    colorChannelNum=xClean.shape[3]
    x = tensorflow.placeholder(tensorflow.float32, shape=(None, imgRows, imgCols, colorChannelNum))
    y = tensorflow.placeholder(tensorflow.float32, shape=(None, yClean.shape[1]))
    modelForIFGSM = KerasModelWrapper(publicModel) #Wrap the keras model to be compatible with Cleverhans 
    fgsm = FastGradientMethod(modelForIFGSM, sess=session)
    preds=modelForIFGSM(x)
    bestScore=1.0
    xAdvTemp=xClean
    xAdvBest=xClean
    for i in range(0,maxIterations):
        fgsm_params = {'eps': eps, 'clip_min': clipMin, 'clip_max':clipMax,'y': yClean}
        xAdv = fgsm.generate(x, **fgsm_params)
        xAdv = tensorflow.stop_gradient(xAdv)
        predsAdv= modelForIFGSM.get_logits(xAdv)
        xAdvTemp=session.run(xAdv, feed_dict={x : xAdvTemp})
        currentScore=publicModel.evaluate(xAdvTemp, yClean, verbose=0)
        print("Iteration ", i)
        print("Current score= ", currentScore[1])
        print("=====")
        if currentScore[1]<bestScore:
            bestScore=currentScore[1]
            xAdvBest=xAdvTemp
    return xAdvBest

#Untargeted Fast Gradient Sign Method (FGSM) using code from Cleverhans: https://github.com/tensorflow/cleverhans
def FGSMNOTarget(session, publicModel,  xClean, yClean, eps, clipMin, clipMax):
    imgRows=xClean.shape[1]
    imgCols=xClean.shape[2]
    colorChannelNum=xClean.shape[3]
    x = tensorflow.placeholder(tensorflow.float32, shape=(None, imgRows, imgCols, colorChannelNum))
    y = tensorflow.placeholder(tensorflow.float32, shape=(None, yClean.shape[1]))
    modelForFGSM = KerasModelWrapper(publicModel) #Wrap the keras model to be compatible with Cleverhans 
    fgsm = FastGradientMethod(modelForFGSM, sess=session)
    preds=modelForFGSM(x)
    fgsm_params = {'eps': eps, 'clip_min': clipMin, 'clip_max':clipMax,'y': yClean}
    xAdv = fgsm.generate(x, **fgsm_params)
    xAdv = tensorflow.stop_gradient(xAdv)
    predsAdv= modelForFGSM.get_logits(xAdv)
    xAdvOut=session.run(xAdv, feed_dict={x : xClean})
    return xAdvOut

def FGSMTargeted(session, publicModel,  xClean, yAdv, eps, clipMin, clipMax):
    imgRows=xClean.shape[1]
    imgCols=xClean.shape[2]
    colorChannelNum=xClean.shape[3]
    x = tensorflow.placeholder(tensorflow.float32, shape=(None, imgRows, imgCols, colorChannelNum))
    y = tensorflow.placeholder(tensorflow.float32, shape=(None, yAdv.shape[1]))
    modelForFGSM = KerasModelWrapper(publicModel) #Wrap the keras model to be compatible with Cleverhans 
    fgsm = FastGradientMethod(modelForFGSM, sess=session)
    preds=modelForFGSM(x)
    fgsm_params = {'eps': eps, 'clip_min': clipMin, 'clip_max':clipMax,'y_target': yAdv}
    xAdv = fgsm.generate(x, **fgsm_params)
    xAdv = tensorflow.stop_gradient(xAdv)
    predsAdv= modelForFGSM.get_logits(xAdv)
    xAdvOut=session.run(xAdv, feed_dict={x : xClean})
    return xAdvOut

#Targeted Project Gradient Descent (PGD) attack with a random noise injected to starting sample
#Uses the l-infinity norm 
def PGDTargeted(session, publicModel, xClean, yAdv, maxIterations, eps, maxChangeEps, rStart, clipMin, clipMax):
    modelForPGD = KerasModelWrapper(publicModel) #Wrap the keras model to be compatible with Cleverhans 
    imgRows=xClean.shape[1]
    imgCols=xClean.shape[2]
    colorChannelNum=xClean.shape[3]
    x = tensorflow.placeholder(tensorflow.float32, shape=(None, imgRows, imgCols, colorChannelNum))
    y = tensorflow.placeholder(tensorflow.float32, shape=(None, yAdv.shape[1]))
    #Choose whether to include rStart as a parameter or not 
    if rStart == 0:
        pgdParams = {'eps':maxChangeEps,'eps_iter': eps, 'clip_min': clipMin, 'clip_max':clipMax, 'y_target': yAdv, 'ord': numpy.inf, 'nb_iter': maxIterations}
    else:
        pgdParams = {'eps':maxChangeEps,'eps_iter': eps, 'clip_min': clipMin, 'clip_max':clipMax, 'y_target': yAdv, 'ord': numpy.inf, 'nb_iter': maxIterations, 'rand_init': rStart}
    pgd = ProjectedGradientDescent(modelForPGD, session)
    xAdv = pgd.generate(x, **pgdParams)
    xAdvFinal = xClean #have to initalize xAdvFinal before using it 
    xAdvFinal=session.run(xAdv, feed_dict={x : xAdvFinal})
    return xAdvFinal

#Untargeted Project Gradient Descent (PGD) attack with a random noise injected to starting sample
#Uses the l-infinity norm 
def PGDNotTargeted(session, publicModel, xClean, yClean, maxIterations, eps, maxChangeEps, rStart, clipMin, clipMax):
    modelForPGD = KerasModelWrapper(publicModel) #Wrap the keras model to be compatible with Cleverhans 
    imgRows=xClean.shape[1]
    imgCols=xClean.shape[2]
    colorChannelNum=xClean.shape[3]
    x = tensorflow.placeholder(tensorflow.float32, shape=(None, imgRows, imgCols, colorChannelNum))
    y = tensorflow.placeholder(tensorflow.float32, shape=(None, yClean.shape[1]))
    #Choose whether to include rStart as a parameter or not 
    if rStart == 0:
        pgdParams = {'eps':maxChangeEps, 'eps_iter': eps, 'clip_min': clipMin, 'clip_max':clipMax, 'y': yClean, 'ord': numpy.inf, 'nb_iter': maxIterations}
    else:
        pgdParams = {'eps':maxChangeEps, 'eps_iter': eps, 'clip_min': clipMin, 'clip_max':clipMax, 'y': yClean, 'ord': numpy.inf, 'nb_iter': maxIterations, 'rand_init': rStart}
    pgd = ProjectedGradientDescent(modelForPGD, session)
    xAdv = pgd.generate(x, **pgdParams)
    xAdvFinal = xClean #have to initalize xAdvFinal before using it 
    xAdvFinal=session.run(xAdv, feed_dict={x : xAdvFinal})
    return xAdvFinal

#Untargeted Momentum Iterative Method 
def MIMNotTargeted(session, publicModel,  xClean, yClean, maxIterations, eps, maxChangeEps, decayFactor, clipMin, clipMax):
    modelForMIM = KerasModelWrapper(publicModel) #Wrap the keras model to be compatible with Cleverhans 
    imgRows=xClean.shape[1]
    imgCols=xClean.shape[2]
    colorChannelNum=xClean.shape[3]
    x = tensorflow.placeholder(tensorflow.float32, shape=(None, imgRows, imgCols, colorChannelNum))
    y = tensorflow.placeholder(tensorflow.float32, shape=(None, yClean.shape[1]))
    mimParams ={'eps':maxChangeEps, 'eps_iter': eps, 'y':yClean, 'clip_min':clipMin, 'clip_max':clipMax, 'nb_iter': maxIterations, 'decay_factor': decayFactor}
    mim = MomentumIterativeMethod(modelForMIM, session)
    xAdv = mim.generate(x, **mimParams)
    xAdvFinal = xClean #have to initalize xAdvFinal before using it 
    xAdvFinal=session.run(xAdv, feed_dict={x : xAdvFinal})
    return xAdvFinal 

#Targeted Momentum Iterative Method
def MIMTargeted(session, publicModel, xClean, yAdv, maxIterations, eps, maxChangeEps, decayFactor, clipMin, clipMax):
    modelForMIM = KerasModelWrapper(publicModel) #Wrap the keras model to be compatible with Cleverhans 
    imgRows=xClean.shape[1]
    imgCols=xClean.shape[2]
    colorChannelNum=xClean.shape[3]
    x = tensorflow.placeholder(tensorflow.float32, shape=(None, imgRows, imgCols, colorChannelNum))
    y = tensorflow.placeholder(tensorflow.float32, shape=(None, yAdv.shape[1]))
    mimParams ={'eps':maxChangeEps, 'eps_iter': eps, 'y_target':yAdv, 'clip_min':clipMin, 'clip_max':clipMax, 'nb_iter': maxIterations, 'decay_factor': decayFactor}
    mim = MomentumIterativeMethod(modelForMIM, session)
    xAdv = mim.generate(x, **mimParams)
    xAdvFinal = xClean #have to initalize xAdvFinal before using it 
    xAdvFinal=session.run(xAdv, feed_dict={x : xAdvFinal})
    return xAdvFinal 

def ElasticNetTargeted(session, publicModel,  xClean, yAdv, beta, batchSize, confidence, learningRate, binarySearchSteps, maxIterations, initialConstant, clipMin, clipMax):
    modelForElasticNet = KerasModelWrapper(publicModel) #Wrap the keras model to be compatible with Cleverhans 
    imgRows=xClean.shape[1]
    imgCols=xClean.shape[2]
    colorChannelNum=xClean.shape[3]
    x = tensorflow.placeholder(tensorflow.float32, shape=(None, imgRows, imgCols, colorChannelNum))
    y = tensorflow.placeholder(tensorflow.float32, shape=(None, yAdv.shape[1]))
    elasticNetParams ={'y_target': yAdv, 'beta': beta, 'decision_rule': 'EN', 'batch_size': batchSize,
        'confidence':confidence,
        'learning_rate':learningRate,
        'binary_search_steps':binarySearchSteps,
        'max_iterations':maxIterations,
        'abort_early':False,
        'initial_const':initialConstant,
        'clip_min':clipMin,
        'clip_max':clipMax}
    ead = ElasticNetMethod(modelForElasticNet, session)
    xAdv = ead.generate(x, **elasticNetParams)
    xAdvFinal = xClean #have to initalize xAdvFinal before using it 
    xAdvFinal=session.run(xAdv, feed_dict={x : xAdvFinal})
    return xAdvFinal 

def ElasticNetNOTargeted(session, publicModel,  xClean, yClean, beta, batchSize, confidence, learningRate, binarySearchSteps, maxIterations, initialConstant, clipMin, clipMax):
    modelForElasticNet = KerasModelWrapper(publicModel) #Wrap the keras model to be compatible with Cleverhans 
    imgRows=xClean.shape[1]
    imgCols=xClean.shape[2]
    colorChannelNum=xClean.shape[3]
    x = tensorflow.placeholder(tensorflow.float32, shape=(None, imgRows, imgCols, colorChannelNum))
    y = tensorflow.placeholder(tensorflow.float32, shape=(None, yClean.shape[1]))
    elasticNetParams ={'y': yClean, 'beta': beta, 'decision_rule': 'EN', 'batch_size': batchSize,
        'confidence':confidence,
        'learning_rate':learningRate,
        'binary_search_steps':binarySearchSteps,
        'max_iterations':maxIterations,
        'abort_early':False,
        'initial_const':initialConstant,
        'clip_min':clipMin,
        'clip_max':clipMax}
    ead = ElasticNetMethod(modelForElasticNet, session)
    xAdv = ead.generate(x, **elasticNetParams)
    xAdvFinal = xClean #have to initalize xAdvFinal before using it 
    xAdvFinal=session.run(xAdv, feed_dict={x : xAdvFinal})
    return xAdvFinal
