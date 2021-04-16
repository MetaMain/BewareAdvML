#Methods for doing the attacks on the vanilla networks CIFAR-10 and Fashion-MNIST
import DataManager
import DefenseMultiModel
import AttackWrappersBlackBox
import tensorflow 
import numpy
from tensorflow import keras
import NetworkConstructors

#Run every attack on the vanilla model 
def RunAllCIFAR10VanillaAttacks():
    for i in range(0,5):
        #Create a new session
        config = tensorflow.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        sess = tensorflow.Session(config=config)
        keras.backend.set_session(sess)
        #construct the model 
        oracleModel = ConstructVanillaCIFAR10Model()
        #Run the attack 
        PapernotAttackCIFAR10(sess, oracleModel, i)
        #End the session 
        sess.close()
        keras.backend.clear_session()

#CIFAR10 Vanilla constructor 
def ConstructVanillaCIFAR10Model():
    classifierDir = "C://Users//Windows//Desktop//Saved Keras Networks//CIFAR10//Vanilia Resnet 56 Aug//cifar10_ResNet6v2_model.162.h5"
    model = DataManager.LoadModel(classifierDir)
    threshold=1
    modelList=[]
    modelList.append(model)
    oracleModel = MultiModel.MultiModel(modelList, 10, threshold)
    return oracleModel

#Run the Papernot attack with a specific amount of starting data 
def PapernotAttackCIFAR10(sess, oracleModel, indexNumber):
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
    saveTag = "PapernotAttack-"+str(queryNum)+"-Vanilla"+"-CIFAR10"
    AttackWrappersBlackBox.PapernotAttackFull(sess, xTrain, yTrain, xTest, yTest, oracleModel, syntheticModel, queryNum, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag)

#Run every attack on the vanilla model 
def RunAllFashionMNISTVanillaAttacks():
    for i in range(0,5):
        #Create a new session
        config = tensorflow.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        sess = tensorflow.Session(config=config)
        keras.backend.set_session(sess)
        #construct the model 
        oracleModel = ConstructFashionMNISTVanillaModel()
        #Run the attack 
        PapernotAttackFashionMNIST(sess, oracleModel, i)
        #End the session 
        sess.close()
        keras.backend.clear_session()

#Fashion MNIST Vanilla constructor 
def ConstructFashionMNISTVanillaModel():
    classifierDir = "C://Users//Windows//Desktop//Saved Keras Networks//Fashion MNIST//Vanillia//VGG16Resize=32ZeroPad=NonePertG=NoneFashionMNISTVanillia.h5"
    model = DataManager.LoadModel(classifierDir)
    threshold=1
    modelList=[]
    modelList.append(model)
    oracleModel = MultiModel.MultiModel(modelList, 10, threshold)
    return oracleModel

def PapernotAttackFashionMNIST(sess, oracleModel, indexNumber):
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
    queryNumArray =  numpy.array([1.0, 0.75, 0.5,  0.25, 0.01]) #Do 100%, 75%, 50%, 25% and 1% attack runs 
    queryNumArray = numpy.round(totalSampleNum * queryNumArray)
    queryNum = int(queryNumArray[indexNumber])
    saveTag = "PapernotAttack-"+str(queryNum)+"-Vanilla"+"-FMNIST"
    #End query amount 
    inputShape=xTrain[0].shape
    numClasses=yTrain.shape[1]
    syntheticModel = NetworkConstructors.ConstructCarliniNetwork(inputShape, numClasses)
    AttackWrappersBlackBox.PapernotAttackFull(sess, xTrain, yTrain, xTest, yTest, oracleModel, syntheticModel, queryNum, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag)

