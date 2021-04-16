#These are the default methods for the ComDefend defense 
import DefenseComDefend
import DataManager
import AttackWrappersBlackBox
import NetworkConstructors
import numpy
import tensorflow 
from tensorflow import keras

#Construct the ComDefend defense for CIFAR10 
def ConstructComDefendCIFAR10():
    classifierDir = "C://Users//Windows//Desktop//Saved Keras Networks//CIFAR10//Vanilia Resnet 56 Aug//cifar10_ResNet6v2_model.162.h5"
    model = DataManager.LoadModel(classifierDir)
    mainDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//ComDefendConvert//checkpoints//"
    dirCom= mainDir + "//enc20_0.0001.npy"
    dirRec= mainDir + "//dec20_0.0001.npy"
    fashion = 0 #Specifies we are not using the FashionMNIST dataset 
    comDefNetwork = DefenseComDefend.DefenseComDefend(model, dirCom, dirRec, fashion) 
    return comDefNetwork

#Run the Papernot attack with a specific amount of starting data 
def PapernotAttackCIFAR10(sess):
    oracleModel = ConstructComDefendCIFAR10()
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
    queryNum = int(queryNumArray[4])
    #End query amount 
    inputShape=xTrain[0].shape
    numClasses=yTrain.shape[1]
    syntheticModel = NetworkConstructors.ConstructCarliniNetwork(inputShape, numClasses)
    saveTag = "PapernotAttack-"+str(queryNum)+"-ComDefend"+"-CIFAR10"
    AttackWrappersBlackBox.PapernotAttackFull(sess, xTrain, yTrain, xTest, yTest, oracleModel, syntheticModel, queryNum, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag)

#Construct the ComDefend defense for Fashion-MNIST 
def ConstructComDefendFashionMNIST():
    classifierDir = "C://Users//Windows//Desktop//Saved Keras Networks//Fashion MNIST//Vanillia//VGG16Resize=32ZeroPad=NonePertG=NoneFashionMNISTVanillia.h5"
    model = DataManager.LoadModel(classifierDir)
    mainDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//ComDefendConvert//checkpoints//"
    dirCom= mainDir + '//enc_mnist.npy'
    dirRec= mainDir + '//dec_mnist.npy'
    fashion = 1 #Specifies we are using the Fashion MNIST dataset 
    comDefNetwork = DefenseComDefend.DefenseComDefend(model, dirCom, dirRec, fashion) 
    return comDefNetwork

def PapernotAttackFashionMNIST(sess):
    oracleModel = ConstructComDefendFashionMNIST()
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
    queryNum = int(queryNumArray[4])
    saveTag = "PapernotAttack-"+str(queryNum)+"-ComDefend"+"-FMNIST"
    #End query amount 
    inputShape=xTrain[0].shape
    numClasses=yTrain.shape[1]
    syntheticModel = NetworkConstructors.ConstructCarliniNetwork(inputShape, numClasses)
    AttackWrappersBlackBox.PapernotAttackFull(sess, xTrain, yTrain, xTest, yTest, oracleModel, syntheticModel, queryNum, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag)

#Pure blackbox methods below this 

#This does the pure black-box attack CIFAR10
def PureBlackBoxRunAttacksOnAllTransformationsCIFAR10():
    saveTagAdvSamples = "CIFAR10PureBlackBoxNoDataAug" #CIFAR-10 pre-generated samples
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//January-31-2020, CIFAR10PureBlackBoxNoDataAug"
    saveTagResults= "PureBlackbox-ComDefend"+"-CIFAR10"
    oracleModel = ConstructComDefendCIFAR10()
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)

#This does every pure black-box attack on transformations T=1, 4, 7 and 10 for BaRT
def PureBlackBoxRunAttacksOnAllTransformationsFashionMNIST():
    saveTagAdvSamples = "FMNISTPureBlackBox" #CIFAR-10 pre-generated samples
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//February-02-2020, FMNISTPureBlackBox"
    oracleModel = ConstructComDefendFashionMNIST()
    saveTagResults= "PureBlackbox-ComDefend"+"-FMNIST"
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)

#def Debug():
#    xTrain, yTrain, xTest, yTest = DataManager.LoadCleanDataCIFAR10()
#    oracleModel = ConstructComDefendCIFAR10()
#    score = oracleModel.evaluate(xTest, yTest)
#    print("Score", score)

#def DebugF():
#    xTrain, yTrain, xTest, yTest = DataManager.LoadCleanDataFashionMNIST()
#    oracleModel = ConstructComDefendFashionMNIST()
#    score = oracleModel.evaluate(xTest, yTest)
#    print("Score", score)
