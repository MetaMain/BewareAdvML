#These are the default methods to run the attack on the Barrage of Random Transforms defense 
import DataManager
import AttackWrappersBlackBox
import NetworkConstructors
import numpy
import tensorflow 
import DefenseBarrageNetwork  
from tensorflow import keras

#This does every mixed black-box attack on transformations T=1, 4, 7 and 10 for BaRT
def RunAttacksOnAllTransformationsCIFAR10():
    transformNumber = [1, 10, 7, 4] 
    for t in range(0, 4): #Go through all the transformations 
        currentTransformNumber = transformNumber[t]
        RunAllCIFAR10AttacksOnBaRT(currentTransformNumber)

def ConstructBaRTDefenseCIFAR10(totalTransformNumber):
    modelDir = "C://Users//Windows//Desktop//Saved Keras Networks//CIFAR10//Barrage Of Random Transforms//cifar10_ResNet6v2_model.123.h5"
    #model = DataManager.LoadModel(dir)
    classNum = 10
    colorChannelNum = 3
    bNet = DefenseBarrageNetwork.DefenseBarrageNetwork(modelDir, totalTransformNumber, classNum, colorChannelNum)
    return bNet

#Run every attack mixed black-box attack on a BaRT model with a given transformation number 
def RunAllCIFAR10AttacksOnBaRT(totalTransformNumber):
    for i in range(0,5):
        #Create a new session
        config = tensorflow.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        sess = tensorflow.Session(config=config)
        keras.backend.set_session(sess)
        #construct the model
        oracleModel = ConstructBaRTDefenseCIFAR10(totalTransformNumber)
        #Run the attack 
        PapernotAttackCIFAR10(sess, oracleModel, totalTransformNumber ,i)
        #End the session 
        sess.close()
        keras.backend.clear_session()

#Run the Papernot attack with a specific amount of starting data 
def PapernotAttackCIFAR10(sess, oracleModel, totalTransformNumber, indexNumber):
    #FGSM
    epsFGSM = 0.05
    #BIM, MIM, PGD
    epsForBIM = 0.005
    maxChangeEpsBIM = 0.05
    rStartPGD = 8.0/255.0 -0.5
    maxIterationsBIM = 10
    decayFactorMIM = 1.0
    #Carlini
    attackBatchSizeCarlini = 1000
    maxIterationsCarlini = 1000
    betaEAD = 0.01
    #Papernot attack parameters
    attackSampleNumber=1000
    batchSizeSynthetic=128
    learning_rate=0.001
    epochForSyntheticModel=10
    #data_aug = 6 #original attack used 6, we don't need to go that high 
    data_aug = 4
    lmbda=0.1
    aug_batch_size= 512 
    clipMin=-0.5
    clipMax=0.5
    xTrain, yTrain, xTest, yTest = DataManager.LoadCleanDataCIFAR10()
    #Pick the query amount
    totalSampleNum = xTrain.shape[0]
    queryNumArray =  numpy.array([1.0, 0.75, 0.5,  0.25, 0.01]) #Do 100%, 75%, 50%, 25% and 1% attack runs 
    queryNumArray = numpy.round(totalSampleNum * queryNumArray)
    queryNum = int(queryNumArray[indexNumber])
    #End query amount 
    inputShape=xTrain[0].shape
    numClasses=yTrain.shape[1]
    syntheticModel = NetworkConstructors.ConstructCarliniNetwork(inputShape, numClasses)
    saveTag = "PapernotAttack-"+str(queryNum)+"-BaRT"+str(totalTransformNumber)+"-CIFAR10"
    AttackWrappersBlackBox.PapernotAttackFull(sess, xTrain, yTrain, xTest, yTest, oracleModel, syntheticModel, queryNum, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag)

def ConstructBaRTDefenseFashionMNIST(totalTransformNumber):
    modelDir = "C://Users//Windows//Desktop//Saved Keras Networks//Fashion MNIST//Barrage Of Random Transforms//VGG16Resize=32ZeroPad=NonePertG=NoneFashionMNISTBarrage.h5"
    #model = DataManager.LoadModel(dir)
    classNum = 10
    colorChannelNum = 1
    bNet = DefenseBarrageNetwork.DefenseBarrageNetwork(modelDir, totalTransformNumber, classNum, colorChannelNum)
    return bNet

def PapernotAttackFashionMNIST(sess, oracleModel, totalTransformNumber, indexNumber):
    #FGSM
    epsFGSM = 0.15
    #BIM, MIM, PGD
    epsForBIM = 0.015
    maxChangeEpsBIM = 0.1
    rStartPGD = 0.15-0.5 #Madry picks 0.3 for all attacks here since we bound to 0.15 that means the random perturbation should also be 0.15
    maxIterationsBIM = 10
    decayFactorMIM = 1.0
    #Carlini
    attackBatchSizeCarlini = 1000
    maxIterationsCarlini = 1000
    betaEAD = 0.01
    #Papernot attack parameters
    queryNum=60000
    attackSampleNumber=1000
    batchSizeSynthetic=128
    learning_rate=0.001
    epochForSyntheticModel=10
    #data_aug = 6 #original attack used 6, we don't need to go that high 
    data_aug = 4
    lmbda=0.1
    aug_batch_size= 512 
    clipMin=-0.5
    clipMax=0.5
    xTrain, yTrain, xTest, yTest = DataManager.LoadCleanDataFashionMNIST()
    #Pick the query amount
    totalSampleNum = xTrain.shape[0]
    queryNumArray =  numpy.array([1.0, 0.75, 0.5,  0.25, 0.01]) #Do 100%, 75%, 50%, 25% or 1% attack runs 
    queryNumArray = numpy.round(totalSampleNum * queryNumArray)
    queryNum = int(queryNumArray[indexNumber])
    saveTag = "PapernotAttack-"+str(queryNum)+"-BaRT"+str(totalTransformNumber)+"-FMNIST"
    #End query amount 
    inputShape=xTrain[0].shape
    numClasses=yTrain.shape[1]
    syntheticModel = NetworkConstructors.ConstructCarliniNetwork(inputShape, numClasses)
    AttackWrappersBlackBox.PapernotAttackFull(sess, xTrain, yTrain, xTest, yTest, oracleModel, syntheticModel, queryNum, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag)

#Run every attack mixed black-box attack on a BaRT model with a given transformation number 
def RunAllFashionMNISTAttacksOnBaRT(totalTransformNumber):
    for i in range(0,5):
        #Create a new session
        config = tensorflow.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        sess = tensorflow.Session(config=config)
        keras.backend.set_session(sess)
        #construct the model
        oracleModel = ConstructBaRTDefenseFashionMNIST(totalTransformNumber)
        #Run the attack 
        PapernotAttackFashionMNIST(sess, oracleModel, totalTransformNumber ,i)
        #End the session 
        sess.close()
        keras.backend.clear_session()

#This does every mixed black-box attack on transformations T=1, 4, 7 and 10 for BaRT
def RunAttacksOnAllTransformationsFashionMNIST():
    #Can only do 1-8 transformations for grayscale (2 transformation groups not applicable)
    transformNumber = [8, 6, 4, 1]
    #for t in range(0, 4): #Go through all the transformations 
    for t in range(0, len(transformNumber)): #Go through all the transformations 
        currentTransformNumber = transformNumber[t]
        RunAllFashionMNISTAttacksOnBaRT(currentTransformNumber)

#Below this is code for pure black box attacks 

#This does every pure black-box attack on transformations T=1, 4, 7 and 10 for BaRT
def PureBlackBoxRunAttacksOnAllTransformationsFashionMNIST():
    transformNumber = [8, 6, 4, 1]
    saveTagAdvSamples = "FMNISTPureBlackBox" #CIFAR-10 pre-generated samples
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//February-02-2020, FMNISTPureBlackBox"
    for t in range(0, 4): #Go through all the transformations 
        #Create new session 
        config = tensorflow.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        sess = tensorflow.Session(config=config)
        keras.backend.set_session(sess)
        #Build the defense 
        currentTransformNumber = transformNumber[t]
        oracleModel = ConstructBaRTDefenseFashionMNIST(currentTransformNumber)       
        saveTagResults= "PureBlackbox-BaRT"+str(currentTransformNumber)+"-FMNIST"
        AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)
        #End the session 
        sess.close()
        keras.backend.clear_session()

#This does every pure black-box attack on transformations T=1, 4, 7 and 10 for BaRT
def PureBlackBoxRunAttacksOnAllTransformationsCIFAR10():
    transformNumber = [1, 10, 7, 4] 
    saveTagAdvSamples = "CIFAR10PureBlackBoxNoDataAug" #CIFAR-10 pre-generated samples
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//January-31-2020, CIFAR10PureBlackBoxNoDataAug"
    for t in range(0, 4): #Go through all the transformations 
        #Create new session 
        config = tensorflow.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        sess = tensorflow.Session(config=config)
        keras.backend.set_session(sess)
        #Build the defense 
        currentTransformNumber = transformNumber[t]
        oracleModel = ConstructBaRTDefenseCIFAR10(currentTransformNumber)       
        saveTagResults= "PureBlackbox-BaRT"+str(currentTransformNumber)+"-CIFAR-10"
        AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)
        #End the session 
        sess.close()
        keras.backend.clear_session()