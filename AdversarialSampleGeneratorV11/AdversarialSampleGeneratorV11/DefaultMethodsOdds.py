import numpy 
import DefenseOddsAreOdd
import tensorflow 
from tensorflow import keras 
import DataManager
import NetworkConstructors
import AttackWrappersBlackBox

def ConstructCIFAR10Def():
    fpr = 40 
    dirModel = "C://Users//Windows//Desktop//Saved Keras Networks//CIFAR10//Vanilia Resnet 56 Aug//cifar10_ResNet6v2_model.162.h5"
    dataset = 'cifar-10'
    mean = numpy.load("C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//OddsAreOddTesterV2//OddsAreOddTesterV2//params_cifar10//mean_yz_training_data_cifar10.npy")                      
    std  = numpy.load("C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//OddsAreOddTesterV2//OddsAreOddTesterV2//params_cifar10//std_yz_training_data_cifar10.npy")
    oracleModel = DefenseOddsAreOdd.DefenseOddsAreOdd(dirModel, "cifar-10", mean, std, fpr)
    return oracleModel

#Run the Papernot attack with a specific amount of starting data 
def PapernotAttackCIFAR10(sess):
    oracleModel = ConstructCIFAR10Def()
    #indexNumber = 0
    #indexNumber = 1
    #indexNumber = 2
    indexNumber = 3
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
    saveTag = "PapernotAttack-"+str(queryNum)+"-OddsDef"+"-CIFAR10"
    AttackWrappersBlackBox.PapernotAttackFull(sess, xTrain, yTrain, xTest, yTest, oracleModel, syntheticModel, queryNum, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag)

def ConstructFashionMNISTDef():
    fpr = 1 
    dirModel = "C://Users//Windows//Desktop//Saved Keras Networks//Fashion MNIST//Vanillia//VGG16Resize=32ZeroPad=NonePertG=NoneFashionMNISTVanillia.h5"
    dataset = 'cifar-10'
    mean = numpy.load("C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//OddsAreOddTesterV2//OddsAreOddTesterV2//params_FashionMNIST//mean_yz_training_data_Fashion_MNIST.npy")                      
    std  = numpy.load("C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//OddsAreOddTesterV2//OddsAreOddTesterV2//params_FashionMNIST//std_yz_training_data_Fashion_MNIST.npy")
    oracleModel = DefenseOddsAreOdd.DefenseOddsAreOdd(dirModel, "fashion-mnist", mean, std, fpr)
    return oracleModel

def PapernotAttackFashionMNIST(sess):
    #indexNumber = 0
    #indexNumber = 1
    #indexNumber = 2
    #indexNumber = 3
    indexNumber = 4
    oracleModel = ConstructFashionMNISTDef()
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
    #queryNum=60000
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
    saveTag = "PapernotAttack-"+str(queryNum)+"-OddsDef"+"-FMNIST"
    #End query amount 
    inputShape=xTrain[0].shape
    numClasses=yTrain.shape[1]
    syntheticModel = NetworkConstructors.ConstructCarliniNetwork(inputShape, numClasses)
    AttackWrappersBlackBox.PapernotAttackFull(sess, xTrain, yTrain, xTest, yTest, oracleModel, syntheticModel, queryNum, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag)

#Pure blackbox methods below here 

#This does the pure black-box attack CIFAR10
def PureBlackBoxRunAttacksOnAllTransformationsCIFAR10():
    saveTagAdvSamples = "CIFAR10PureBlackBoxNoDataAug" #CIFAR-10 pre-generated samples
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//January-31-2020, CIFAR10PureBlackBoxNoDataAug"
    saveTagResults= "PureBlackbox-OddsDef"+"-CIFAR10"
    oracleModel = ConstructCIFAR10Def()
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)
       
#This does every pure black-box attack on transformations T=1, 4, 7 and 10 for BaRT
def PureBlackBoxRunAttacksOnAllTransformationsFashionMNIST():
    saveTagAdvSamples = "FMNISTPureBlackBox" #CIFAR-10 pre-generated samples
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//February-02-2020, FMNISTPureBlackBox"
    oracleModel = ConstructFashionMNISTDef()
    saveTagResults= "PureBlackbox-OddsDef"+"-FMNIST"
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)