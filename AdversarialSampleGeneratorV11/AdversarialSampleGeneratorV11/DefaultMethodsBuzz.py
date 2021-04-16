#From the Buzz paper 
import DataManager
import AttackWrappersBlackBox
import NetworkConstructors
import numpy
import tensorflow 
import DefenseMultiModel
from tensorflow import keras

#Run every attack on the BUZZ model 
def RunAllCIFAR10BUZZEightAttacks():
    for i in range(0,5):
        #Create a new session
        config = tensorflow.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        sess = tensorflow.Session(config=config)
        keras.backend.set_session(sess)
        #construct the model 
        baseDir = "C://Users//Windows//Desktop//Saved Keras Networks//CIFAR10//Hateful 8 Resnet Data Aug V0"
        oracleModel = ConstructBUZZResnet8(baseDir)
        #Run the attack 
        PapernotAttackCIFAR10(sess, oracleModel, i)
        #End the session 
        sess.close()
        keras.backend.clear_session()

#Run every attack on the BUZZ-2 model 
def RunAllCIFAR10BUZZTwoAttacks():
    for i in range(0,5):
        #Create a new session
        config = tensorflow.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        sess = tensorflow.Session(config=config)
        keras.backend.set_session(sess)
        #construct the model 
        baseDir = "C://Users//Windows//Desktop//Saved Keras Networks//CIFAR10//Hateful 8 Resnet Data Aug V0"
        oracleModel = ConstructBUZZResnet2(baseDir)
        #Run the attack 
        PapernotAttackCIFAR10(sess, oracleModel, i)
        #End the session 
        sess.close()
        keras.backend.clear_session()

def ConstructBUZZResnet2(baseDir):
    threshold=2
    modelList=[]
    modelList.append(DataManager.LoadModel(baseDir+"//BUZZ32_ResNet6v2_model.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//BUZZ104_ResNet6v2_model.h5"))
    mm=DefenseMultiModel.DefenseMultiModel(modelList, 10, threshold)
    return mm

def ConstructBUZZResnet8(baseDir):
    threshold=8
    modelList=[]
    modelList.append(DataManager.LoadModel(baseDir+"//BUZZ32_ResNet6v2_model.h5"))
    print("sanity check")
    modelList.append(DataManager.LoadModel(baseDir+"//BUZZ40_ResNet6v2_model.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//BUZZ48_ResNet6v2_model.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//BUZZ64_ResNet6v2_model.h5"))
    print("sanity check")
    modelList.append(DataManager.LoadModel(baseDir+"//BUZZ72_ResNet6v2_model.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//BUZZ80_ResNet6v2_model.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//BUZZ96_ResNet6v2_model.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//BUZZ104_ResNet6v2_model.h5"))
    print("sanity check")
    mm=DefenseMultiModel.DefenseMultiModel(modelList, 10, threshold)
    return mm

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
    saveTag = "PapernotAttack-"+str(queryNum)+"-BUZZ8"+"-CIFAR10"
    AttackWrappersBlackBox.PapernotAttackFull(sess, xTrain, yTrain, xTest, yTest, oracleModel, syntheticModel, queryNum, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag)

#This does the pure black-box attack CIFAR10
def PureBlackBoxRunAttacksBUZZ2CIFAR10():
    saveTagAdvSamples = "CIFAR10PureBlackBoxNoDataAug" #CIFAR-10 pre-generated samples
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//January-31-2020, CIFAR10PureBlackBoxNoDataAug"
    saveTagResults= "PureBlackbox-BUZZ2"+"-CIFAR10"
    baseDir = "C://Users//Windows//Desktop//Saved Keras Networks//CIFAR10//Hateful 8 Resnet Data Aug V0"
    oracleModel = ConstructBUZZResnet2(baseDir)
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)

def PureBlackBoxRunAttacksBUZZ8CIFAR10():
    saveTagAdvSamples = "CIFAR10PureBlackBoxNoDataAug" #CIFAR-10 pre-generated samples
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//January-31-2020, CIFAR10PureBlackBoxNoDataAug"
    saveTagResults= "PureBlackbox-BUZZ8"+"-CIFAR10"
    baseDir = "C://Users//Windows//Desktop//Saved Keras Networks//CIFAR10//Hateful 8 Resnet Data Aug V0"
    oracleModel = ConstructBUZZResnet8(baseDir)
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)
