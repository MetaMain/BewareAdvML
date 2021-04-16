#These are the default methods for BUZZ FashionMNIST
import DataManager
import NetworkConstructors
import AttackWrappersBlackBox
import DefenseMultiModel 
import tensorflow 
from tensorflow import keras
import numpy

#Run every attack on the BUZZ-2 model 
def RunAllFmnistBUZZEightAttacks():
    saveTag = "BUZZ8"
    for i in range(0,5):
        #Create a new session
        config = tensorflow.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        sess = tensorflow.Session(config=config)
        keras.backend.set_session(sess)
        #construct the model 
        oracleModel = ConstructVGGBUZZ8()
        #Run the attack 
        PapernotAttackFashionMNIST(sess, oracleModel, i, saveTag)
        #End the session 
        sess.close()
        keras.backend.clear_session()

#Run every attack on the BUZZ-2 model 
def RunAllFmnistBUZZTwoAttacks():
    saveTag = "BUZZ2"
    for i in range(0,5):
        #Create a new session
        config = tensorflow.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth=True
        sess = tensorflow.Session(config=config)
        keras.backend.set_session(sess)
        #construct the model 
        oracleModel = ConstructVGGBUZZ2()
        #Run the attack 
        PapernotAttackFashionMNIST(sess, oracleModel, i, saveTag)
        #End the session 
        sess.close()
        keras.backend.clear_session()

 #This is for FashionMNIST
def ConstructVGGBUZZ8():
    baseDir = "C://Users//Windows//Desktop//Saved Keras Networks//Fashion MNIST//Eprint Hateful V1" 
    threshold=8
    modelList=[]
    modelList.append(DataManager.LoadModel(baseDir+"//VGG16Resize=32ZeroPad=NonePertG=NoneFashionMNISTAandB.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//VGG16Resize=40ZeroPad=NonePertG=NoneFashionMNISTAandB.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//VGG16Resize=48ZeroPad=NonePertG=NoneFashionMNISTAandB.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//VGG16Resize=64ZeroPad=NonePertG=NoneFashionMNISTAandB.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//VGG16Resize=72ZeroPad=NonePertG=NoneFashionMNISTAandB.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//VGG16Resize=80ZeroPad=NonePertG=NoneFashionMNISTAandB.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//VGG16Resize=96ZeroPad=NonePertG=NoneFashionMNISTAandB.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//VGG16Resize=104ZeroPad=NonePertG=NoneFashionMNISTAandB.h5"))
    mm=DefenseMultiModel.DefenseMultiModel(modelList, 10, threshold)
    return mm

def ConstructVGGBUZZ2():
    baseDir = "C://Users//Windows//Desktop//Saved Keras Networks//Fashion MNIST//Eprint Hateful V1"
    threshold=2
    modelList=[]
    modelList.append(DataManager.LoadModel(baseDir+"//VGG16Resize=32ZeroPad=NonePertG=NoneFashionMNISTAandB.h5"))
    modelList.append(DataManager.LoadModel(baseDir+"//VGG16Resize=104ZeroPad=NonePertG=NoneFashionMNISTAandB.h5"))
    mm=DefenseMultiModel.DefenseMultiModel(modelList, 10, threshold)
    return mm

def PapernotAttackFashionMNIST(sess, oracleModel, indexNumber, saveTag):
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
    saveTag = "PapernotAttack-"+str(queryNum)+"-"+saveTag+"-FMNIST"
    #End query amount 
    inputShape=xTrain[0].shape
    numClasses=yTrain.shape[1]
    syntheticModel = NetworkConstructors.ConstructCarliniNetwork(inputShape, numClasses)
    AttackWrappersBlackBox.PapernotAttackFull(sess, xTrain, yTrain, xTest, yTest, oracleModel, syntheticModel, queryNum, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag)

#This does the pure black-box attack
def PureBlackBoxRunAttacksBUZZ2FashionMNIST():
    saveTagAdvSamples = "FMNISTPureBlackBox" #CIFAR-10 pre-generated samples
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//February-02-2020, FMNISTPureBlackBox"
    oracleModel = ConstructVGGBUZZ2()
    saveTagResults= "PureBlackbox-BUZZ2"+"-FMNIST"
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)

def PureBlackBoxRunAttacksBUZZ8FashionMNIST():
    saveTagAdvSamples = "FMNISTPureBlackBox" #CIFAR-10 pre-generated samples
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//February-02-2020, FMNISTPureBlackBox"
    oracleModel = ConstructVGGBUZZ8()
    saveTagResults= "PureBlackbox-BUZZ8"+"-FMNIST"
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)