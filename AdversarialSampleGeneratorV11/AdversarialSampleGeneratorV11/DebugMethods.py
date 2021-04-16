import DataManager
import DefenseADPNetwork
import AttackWrappersBlackBox
import DefenseMultiModel
def TestCleanAcc():
    xTrain, yTrain, xTest, yTest = DataManager.LoadCleanDataCIFAR10()
    dir = "C://Users/Windows//Desktop//Saved Keras Networks//CIFAR10//Adaptive Diversit Promoting (ADP)//model.184.h5"
    lambdaVal = 2
    log_delta_lambda = 0.5 
    adp = DefenseADPNetwork.DefenseADPNetwork(dir, "cifar-10", 3, lambdaVal, log_delta_lambda)
    score = adp.evaluate(xTest, yTest)
    print("Clean score:", score)

    #xTrain, yTrain, xTest, yTest = DataManager.LoadCleanDataFashionMNIST()
    #dir = "C://Users//Windows//Desktop//Saved Keras Networks//Fashion MNIST//Adaptive Diversity Promoting (ADP)//ADPVGG16-3Models.h5"
    #lambdaVal = 2
    #log_delta_lambda = 0.5 
    #adp = DefenseADPNetwork.DefenseADPNetwork(dir, "fashion-mnist", 3, lambdaVal, log_delta_lambda)
    #score = adp.evaluate(xTest, yTest)
    #print("Clean score:", score)


#CIFAR-10 Test 
def TestModel():
    dir = "C://Users//Windows//Desktop//Saved Keras Networks//CIFAR10//Model for pure blackbox attack//SyntheticModelNoDataAug300Epochs.h5"
    xTrain, yTrain, xTest, yTest = DataManager.LoadCleanDataCIFAR10()
    syntheticModel = DataManager.LoadModel(dir)
    secretModelList =[]
    secretModelList.append(syntheticModel)
    oracleModel = DefenseMultiModel.DefenseMultiModel(secretModelList, 10, 1)
    #score = model.evaluate(xTest, yTest)
    #print("Clean test score:", score[1])
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//January-31-2020, CIFAR10PureBlackBoxNoDataAug"
    saveTagAdvSamples = "CIFAR10PureBlackBoxNoDataAug"
    saveTagResults= "Debug-Pure-Blackbox-SyntheticModel"
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)
    #Vanilla attack 
    dir = "C://Users//Windows//Desktop//Saved Keras Networks//CIFAR10//Vanilia Resnet 56 Aug//cifar10_ResNet6v2_model.162.h5"
    vanillaModel = DataManager.LoadModel(dir)
    secretModelList =[]
    secretModelList.append(vanillaModel)
    oracleModel = DefenseMultiModel.DefenseMultiModel(secretModelList, 10, 1)
    saveTagResults = "Debug-Pure-Blackbox-VanillaModel"
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)

#Fashion-MNIST Test 
def TestModelFMNIST():
    dir = "C://Users//Windows//Desktop//Saved Keras Networks//Fashion MNIST//Model for pure blackbox attack//SyntheticModelFMNIST-100Epochs.h5"
    xTrain, yTrain, xTest, yTest = DataManager.LoadCleanDataFashionMNIST()
    syntheticModel = DataManager.LoadModel(dir)
    secretModelList =[]
    secretModelList.append(syntheticModel)
    oracleModel = DefenseMultiModel.DefenseMultiModel(secretModelList, 10, 1)
    #score = model.evaluate(xTest, yTest)
    #print("Clean test score:", score[1])
    advDir = "C://Users//Windows//Desktop//Kaleel//Adversarial Neural Network Work 2020//AdversarialSampleGeneratorV11//AdversarialSampleGeneratorV11//February-02-2020, FMNISTPureBlackBox"
    saveTagAdvSamples = "FMNISTPureBlackBox"
    saveTagResults= "Debug-Pure-Blackbox-SyntheticModel-FMNIST"
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)
    #Vanilla attack 
    dir = "C://Users//Windows//Desktop//Saved Keras Networks//Fashion MNIST//Vanillia//VGG16Resize=32ZeroPad=NonePertG=NoneFashionMNISTVanillia.h5"
    vanillaModel = DataManager.LoadModel(dir)
    secretModelList =[]
    secretModelList.append(vanillaModel)
    oracleModel = DefenseMultiModel.DefenseMultiModel(secretModelList, 10, 1)
    saveTagResults = "Debug-Pure-Blackbox-VanillaModel-FMNIST"
    AttackWrappersBlackBox.RunPureBlackBoxAttack(advDir, oracleModel, saveTagAdvSamples, saveTagResults)