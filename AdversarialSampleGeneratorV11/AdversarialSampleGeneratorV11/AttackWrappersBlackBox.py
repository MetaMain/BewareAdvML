import tensorflow
from tensorflow.keras import backend as K
import numpy
import random 
import AttackWrappersWhiteBox
import DataManager
import matplotlib.pyplot as plt
import NetworkConstructors
from cleverhans.attacks import FastGradientMethod
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.loss import CrossEntropy
from cleverhans.utils_tf import jacobian_graph, jacobian_augmentation
from datetime import date
import os 

#This runs the Papernot attack 
#First a synthetic model is trained on the provided and additional synthetic data is created for training 
#Then attacks are run on the synthetic model and the transferability is checked on the oracle model 
def PapernotAttackFull(sess, xTrain, yTrain, xTest, yTest, oracleModel, syntheticModel, queryNum, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag):
    #Create place to save all files
    today = date.today()
    dateString = today.strftime("%B"+"-"+"%d"+"-"+"%Y, ") #Get the year, month, day
    experimentDateAndName = dateString + saveTag #Name of experiment with data 
    saveDir = os.path.join(os.getcwd(), experimentDateAndName)
    if not os.path.isdir(saveDir): #If not there, make the directory 
        os.makedirs(saveDir)
    #Place to save the results 
    os.chdir(saveDir)
    resultsTextFile = open(experimentDateAndName+", Results.txt","a+")

    #First train the synthetic network 
    nb_classes=yTrain.shape[1]
    xSub=xTrain[:queryNum]
    xSub, ySub = FilterData(xSub, oracleModel, nb_classes)

    trainedSyntheticNetwork = TrainSyntheticNetworkThresholdVersion(sess, oracleModel, syntheticModel, xSub, ySub, data_aug, batchSizeSynthetic, epochForSyntheticModel, lmbda, aug_batch_size)
    trainedSyntheticNetwork.save("SyntheticModel"+saveTag+".h5")

    #Synthetic Network has been trained, now time to run all the attacks 
    print("Running all attacks...")
    xClean, yClean = GetCleanCorrectlyIdentifiedSamples(oracleModel, xTest, yTest, attackSampleNumber)
    yAdvTarget = GenerateTargetsLabelRandomly(yClean, nb_classes) #These are the labels used in all the target attacks

    #If the synthetic model has drop out or batch norm AND the defense is PyTorch this will cause attacks to fail for some reason 
    #With the following error: https://github.com/tensorflow/tensorflow/issues/23166
    #Only work around right now is to set the learning phase of keras 
    K.set_learning_phase(0) 

    #FGSM Targeted 
    xAdvFGSMTarget = AttackWrappersWhiteBox.FGSMTargeted(sess, trainedSyntheticNetwork, xClean, yAdvTarget, epsFGSM, clipMin, clipMax)
    #Here the attack works if the oracle outputs the target class label 
    fgsmTargetScore = oracleModel.evaluate(xAdvFGSMTarget, yAdvTarget)
    defenseAccuracyFGSMTargeted = 1 - fgsmTargetScore
    print("Targeted FGSM defense accuracy:", defenseAccuracyFGSMTargeted)
    resultsTextFile.write("Targeted FGSM defense accuracy:"+str(defenseAccuracyFGSMTargeted)+"\n")
    DataManager.SaveAdversarialData("FGSMTargeted"+saveTag, xClean, yClean, xAdvFGSMTarget, yAdvTarget)

    #BIM targeted 
    xAdvIFGSMTargeted = AttackWrappersWhiteBox.PGDTargeted(sess, trainedSyntheticNetwork, xClean, yAdvTarget, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, 0, clipMin, clipMax)
    ifgsmTargetScore = oracleModel.evaluate(xAdvIFGSMTargeted, yAdvTarget)
    defenseAccuracyIFGSMTargeted = 1 - ifgsmTargetScore
    print("Targeted IFGSM defense accuracy:", defenseAccuracyIFGSMTargeted)
    resultsTextFile.write("Targeted IFGSM defense accuracy:"+str(defenseAccuracyIFGSMTargeted)+"\n")
    DataManager.SaveAdversarialData("IFGSMTargeted"+saveTag, xClean, yClean, xAdvIFGSMTargeted)

    #MIM targeted 
    xAdvMIMTargeted = AttackWrappersWhiteBox.MIMTargeted(sess, trainedSyntheticNetwork, xClean, yAdvTarget, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, decayFactorMIM, clipMin, clipMax)
    xAdvMIMTargetedScore = oracleModel.evaluate(xAdvMIMTargeted, yAdvTarget)
    defenseAccuracyMIMTargeted = 1 - xAdvMIMTargetedScore
    print("Targeted MIM defense accuracy:", defenseAccuracyMIMTargeted)
    resultsTextFile.write("Targeted MIM defense accuracy:"+str(defenseAccuracyMIMTargeted)+"\n")
    DataManager.SaveAdversarialData("MIMTargeted"+saveTag, xClean, yClean, xAdvMIMTargeted)

    #PGD targeted 
    xAdvPGDTargeted = AttackWrappersWhiteBox.PGDTargeted(sess, trainedSyntheticNetwork, xClean, yAdvTarget, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, clipMin, clipMax)
    xAdvPGDTargetScore = oracleModel.evaluate(xAdvPGDTargeted, yAdvTarget)
    defenseAccuracyPGDTargeted = 1 - xAdvPGDTargetScore
    print("Targeted PGD defense accuracy:", defenseAccuracyPGDTargeted)
    resultsTextFile.write("Targeted PGD defense accuracy:"+str(defenseAccuracyPGDTargeted)+"\n")
    DataManager.SaveAdversarialData("PGDTargeted"+saveTag, xClean, yClean, xAdvPGDTargeted)

    #FGSM untargeted 
    xAdvFGSMNOTarget = AttackWrappersWhiteBox.FGSMNOTarget(sess, trainedSyntheticNetwork,  xClean, yClean, epsFGSM, clipMin, clipMax)
    #Here the attack works if the oracle outputs the wrong class label AND it is not the noise class label 
    fgsmNOTargetScore = oracleModel.evaluateAdversarialAttackSuccessRate(xAdvFGSMNOTarget, yClean)
    defenseAccuracyFGSMNOTargeted = 1 - fgsmNOTargetScore
    print("Untargeted FGSM defense accuracy:", defenseAccuracyFGSMNOTargeted)
    resultsTextFile.write("Untargeted FGSM defense accuracy:"+str(defenseAccuracyFGSMNOTargeted)+"\n")
    DataManager.SaveAdversarialData("FGSMNOTargeted"+saveTag, xClean, yClean, xAdvFGSMNOTarget)

    #BIM untargeted 
    xAdvIFGSMNOTargeted = AttackWrappersWhiteBox.PGDNotTargeted(sess, trainedSyntheticNetwork, xClean, yClean, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, 0, clipMin, clipMax)
    ifgsmNOTargetScore = oracleModel.evaluateAdversarialAttackSuccessRate(xAdvIFGSMNOTargeted, yClean)
    defenseAccuracyIFGSMNOTargeted = 1 - ifgsmNOTargetScore
    print("Untargeted IFGSM defense accuracy:", defenseAccuracyIFGSMNOTargeted)
    resultsTextFile.write("Untargeted IFGSM defense accuracy:"+str(defenseAccuracyIFGSMNOTargeted)+"\n")
    DataManager.SaveAdversarialData("IFGSMNOTargeted"+saveTag, xClean, yClean, xAdvIFGSMNOTargeted)

    #MIM untargeted
    xAdvMIMNOTarget = AttackWrappersWhiteBox.MIMNotTargeted(sess, trainedSyntheticNetwork,  xClean, yClean, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, decayFactorMIM, clipMin, clipMax)
    xAdvMIMNOTargetedScore = oracleModel.evaluateAdversarialAttackSuccessRate(xAdvMIMNOTarget, yClean)
    defenseAccuracyMIMNOTargeted = 1 - xAdvMIMNOTargetedScore
    print("Untargeted MIM defense accuracy:", defenseAccuracyMIMNOTargeted)
    resultsTextFile.write("Untargeted MIM defense accuracy:"+str(defenseAccuracyMIMNOTargeted)+"\n")
    DataManager.SaveAdversarialData("MIMNOTargeted"+saveTag, xClean, yClean, xAdvMIMNOTarget)

    #PGD untargeted 
    xAdvPGDNOTargeted = AttackWrappersWhiteBox.PGDNotTargeted(sess, trainedSyntheticNetwork, xClean, yClean, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, clipMin, clipMax)
    xAdvPGDNOTargetedScore = oracleModel.evaluateAdversarialAttackSuccessRate(xAdvPGDNOTargeted, yClean)
    defenseAccuracyPGDNOTargeted = 1 - xAdvPGDNOTargetedScore
    print("Untargeted PGD defense accuracy:", defenseAccuracyPGDNOTargeted)
    resultsTextFile.write("Untargeted PGD defense accuracy:"+str(defenseAccuracyPGDNOTargeted)+"\n")
    DataManager.SaveAdversarialData("PGDNOTargeted"+saveTag, xClean, yClean, xAdvPGDNOTargeted)

    #Carlini L2 Targeted 
    xAdvCarliniTargeted = AttackWrappersWhiteBox.CarliniL2Targeted(sess, attackBatchSizeCarlini, maxIterationsCarlini, trainedSyntheticNetwork, xClean, yAdvTarget, clipMin, clipMax)
    xAdvCarliniTargetedScore = oracleModel.evaluate(xAdvCarliniTargeted, yAdvTarget)
    defenseAccuracyCarliniTargeted = 1 - xAdvCarliniTargetedScore
    print("Targeted Carlini defense accuracy:", defenseAccuracyCarliniTargeted)
    resultsTextFile.write("Targeted Carlini defense accuracy:"+str(defenseAccuracyCarliniTargeted)+"\n")
    DataManager.SaveAdversarialData("CarliniL2Targeted"+saveTag, xClean, yClean, xAdvCarliniTargeted)

    #Carlini L2 Untargeted 
    xAdvCarliniNOTargeted = AttackWrappersWhiteBox.CarliniL2NOTarget(sess, attackBatchSizeCarlini, maxIterationsCarlini, trainedSyntheticNetwork, xClean, yClean, clipMin, clipMax)
    xAdvCarliniNOTargetedScore = oracleModel.evaluateAdversarialAttackSuccessRate(xAdvCarliniNOTargeted, yClean)
    defenseAccuracyCarliniNOTargeted = 1 - xAdvCarliniNOTargetedScore
    print("Untargeted Carlini defense accuracy:", defenseAccuracyCarliniNOTargeted)
    resultsTextFile.write("Untargeted Carlini defense accuracy:"+str(defenseAccuracyCarliniNOTargeted)+"\n")
    DataManager.SaveAdversarialData("CarliniL2Untargeted"+saveTag, xClean, yClean, xAdvCarliniNOTargeted)

    #EAD Targeted 
    #We use the Carlini attack parameters again
    confidence = 0.0
    learningRate = 1e-2
    binarySearchSteps = 9
    initialConstant = 1e-3
    xAdvEADTargeted = AttackWrappersWhiteBox.ElasticNetTargeted(sess, trainedSyntheticNetwork, xClean, yAdvTarget, betaEAD, attackBatchSizeCarlini, confidence, learningRate, binarySearchSteps, maxIterationsCarlini, initialConstant, clipMin, clipMax)
    xAdvEADTargetedScore = oracleModel.evaluate(xAdvEADTargeted, yAdvTarget)
    defenseAccuracyEADTargeted = 1 - xAdvEADTargetedScore
    print("Targeted EAD defense accuracy:", defenseAccuracyEADTargeted)
    resultsTextFile.write("Targeted EAD defense accuracy:"+str(defenseAccuracyEADTargeted)+"\n")
    DataManager.SaveAdversarialData("EADTargeted"+saveTag, xClean, yClean, xAdvEADTargeted)

    #EAD Untargeted 
    xAdvEADNOTarget = AttackWrappersWhiteBox.ElasticNetNOTargeted(sess, trainedSyntheticNetwork,  xClean, yClean, betaEAD, attackBatchSizeCarlini, confidence, learningRate, binarySearchSteps, maxIterationsCarlini, initialConstant, clipMin, clipMax)
    xAdvEADNOTargetScore = oracleModel.evaluateAdversarialAttackSuccessRate(xAdvEADNOTarget, yClean)
    defenseAccuracyEADNOTargeted = 1 - xAdvEADNOTargetScore
    print("Untargeted EAD defense accuracy:", defenseAccuracyEADNOTargeted)
    resultsTextFile.write("Untargeted EAD defense accuracy:"+str(defenseAccuracyEADNOTargeted)+"\n")
    DataManager.SaveAdversarialData("EADUntargeted"+saveTag, xClean, yClean, xAdvEADNOTarget)

    #Check the clean accuracy 
    cleanScore = oracleModel.evaluate(xTest, yTest)
    print("The clean score is:", cleanScore)
    resultsTextFile.write("The clean score is:"+str(cleanScore)+"\n")
    resultsTextFile.close() #Close the results file at the end 
    os.chdir("..") #move up one directory to return to original directory 

#In this version the blackbox simply ignores samples which output the noise class label and doesn't use them in training 
def TrainSyntheticNetworkThresholdVersion(sess, oracleModel, syntheticModel, xSub, ySub, data_aug, batchSize, epochForSyntheticModel, lmbda, aug_batch_size):
    img_rows=xSub.shape[1]
    img_cols=xSub.shape[2]
    nchannels=xSub.shape[3]
    nb_classes=ySub.shape[1]
    x = tensorflow.placeholder(tensorflow.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tensorflow.placeholder(tensorflow.float32, shape=(None, nb_classes))

    model_sub = KerasModelWrapper(syntheticModel) #Wrap the keras model to be compatible with Cleverhans 
    preds_sub = model_sub.get_logits(x)
    loss_sub = CrossEntropy(model_sub, smoothing=0)
    # Define the Jacobian symbolically using TensorFlow
    grads = jacobian_graph(preds_sub, x, nb_classes)
    for rho in range(0, data_aug):
        print("Substitute training epoch #" + str(rho))

        syntheticModel.fit(xSub, ySub, batch_size=batchSize, epochs=epochForSyntheticModel, verbose=1)
        #Refresh the computational graph for the jacobian methods 
        model_sub = KerasModelWrapper(syntheticModel) #Wrap the keras model to be compatible with Cleverhans 
        preds_sub = model_sub.get_logits(x)
        loss_sub = CrossEntropy(model_sub, smoothing=0)

        # If we are not at last substitute training iteration, augment dataset
        if rho < data_aug - 1:
            # Perform the Jacobian augmentation
            lmbda_coef = 2 * int(int(rho / 3) != 0) - 1
            ySubTemp=ySub.argmax(axis=1)
            xSub = jacobian_augmentation(sess, x, xSub, ySubTemp, grads, lmbda_coef * lmbda, aug_batch_size)
            print("Labeling substitute training data.")
            xSub, ySub = FilterData(xSub, oracleModel, nb_classes) #Make sure we don't use data labeled with the noise class
    return syntheticModel

#Filter out any data that has the noise class label 
#This is the faster version that uses numpy arrays 
def FilterData(xSub, oracleModel, nb_classes):
    originalSampleSize=xSub.shape[0]
    imgRows=xSub.shape[1]
    imgCols=xSub.shape[2]
    colorChannelNum=xSub.shape[3]
    yPred=oracleModel.predict(xSub)
    yPredInt=yPred.argmax(axis=1) #Convert categorical to int class labels 
    #print(yPredInt[500:800])
    #First get a count of the number of samples that have the noise class label in the training data 
    noiseClassLabel=nb_classes #The last class label is always the noise label
    #First make sure that some of the data has noise class label, if it does not then we can just do normal predict
    #Note this will happen if there is only a single model in your multimodel
    if noiseClassLabel in yPredInt:  
        unique, counts = numpy.unique(yPredInt, return_counts=True)
        numberOfNoiseSamples=counts[nb_classes]
        finalSampleSize=int(xSub.shape[0]-numberOfNoiseSamples)
        xSubFinal=numpy.zeros((finalSampleSize, imgRows, imgCols, colorChannelNum))
        ySubFinal=numpy.zeros((finalSampleSize, nb_classes))
        currentSampleIndex=0
        for i in range(0, originalSampleSize): 
            if yPredInt[i] != nb_classes: #Not a noise sample so we need to store it 
                classLabelIndex=int(yPredInt[i])
                ySubFinal[currentSampleIndex, classLabelIndex]=1.0
                xSubFinal[currentSampleIndex,:,:,:]=xSub[i,:,:,:]
                currentSampleIndex=currentSampleIndex+1
    else: #There are no noise class labels OR a single model is being used so no noise class labels 
        xSubFinal=xSub
        ySubFinal=numpy.zeros((xSub.shape[0], nb_classes))
        for i in range(0, ySubFinal.shape[0]):
            correctIndex=int(yPredInt[i])
            ySubFinal[i,correctIndex]=1.0
    #Return the final numpy arrays
    return xSubFinal, ySubFinal 

#returns a subset, xClean and yClean 
def GetCleanCorrectlyIdentifiedSamples(model, xData, yData, attackSampleNum):
    imgRows=xData.shape[1]
    imgCols=xData.shape[2]
    colorChannelNum=xData.shape[3]
    classNum=yData.shape[1]
    xClean=numpy.zeros((attackSampleNum, imgRows, imgCols, colorChannelNum))
    yClean=numpy.zeros((attackSampleNum,classNum))
    publicTestOutput=model.predict(xData)
    counterA=0
    counterB=0 #The index for the 
    while counterB<attackSampleNum: #Get 1000 correctly classified samples 
        predictedLabelPublic=publicTestOutput[counterA].argmax(axis=0) #Get the public output (as single value)
        yTrueLabel=yData[counterA].argmax(axis=0) #Get the ground truth label (as single value)
        if predictedLabelPublic==yTrueLabel: #The sample is correctly predicted
            xClean[counterB]=xData[counterA]
            yClean[counterB,yTrueLabel]=1.0
            counterB=counterB+1
        counterA=counterA+1
    return xClean, yClean

#This method randomly creates fake labels for the attack 
#The fake target is guaranteed to not be the same as the original class label 
def GenerateTargetsLabelRandomly(yData, numClasses):
    fTargetLabels=numpy.zeros((len(yData),numClasses))
    for i in range(0, len(yData)):
        targetLabel=random.randint(0,numClasses-1)
        trueLabel=yData[i].argmax(axis=0)
        while targetLabel==trueLabel:#Target and true label should not be the same 
            targetLabel=random.randint(0,numClasses-1) #Keep flipping until a different label is achieved 
        fTargetLabels[i,targetLabel]=1.0
    return fTargetLabels

def GenerateAdversarialSamplesForPureBlackBoxAttack(sess, trainedSyntheticNetwork, xTrain, yTrain, xTest, yTest, attackSampleNumber, epsFGSM, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, decayFactorMIM, attackBatchSizeCarlini, maxIterationsCarlini, betaEAD, clipMin, clipMax, saveTag):
    #Create place to save all files
    today = date.today()
    dateString = today.strftime("%B"+"-"+"%d"+"-"+"%Y, ") #Get the year, month, day
    experimentDateAndName = dateString + saveTag #Name of experiment with data 
    saveDir = os.path.join(os.getcwd(), experimentDateAndName)
    if not os.path.isdir(saveDir): #If not there, make the directory 
        os.makedirs(saveDir)
    #Place to save the results 
    os.chdir(saveDir)

    #Synthetic Network has been trained, now time to run all the attacks 
    print("Running all attacks...")
    xClean = xTest[0:attackSampleNumber] 
    yClean = yTest[0:attackSampleNumber] 
    nb_classes = yTest[0].shape[0]
    yAdvTarget = GenerateTargetsLabelRandomly(yClean, nb_classes) #These are the labels used in all the target attacks

    #FGSM Targeted 
    xAdvFGSMTarget = AttackWrappersWhiteBox.FGSMTargeted(sess, trainedSyntheticNetwork, xClean, yAdvTarget, epsFGSM, clipMin, clipMax)
    fgsmTargetScore = trainedSyntheticNetwork.evaluate(xAdvFGSMTarget, yAdvTarget, verbose=2)
    print("Target FGSM attack score:", fgsmTargetScore[1])
    DataManager.SaveAdversarialData("FGSMTargeted"+saveTag, xClean, yClean, xAdvFGSMTarget, yAdvTarget)

    #BIM targeted 
    xAdvIFGSMTargeted = AttackWrappersWhiteBox.PGDTargeted(sess, trainedSyntheticNetwork, xClean, yAdvTarget, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, 0, clipMin, clipMax)
    ifgsmTargetScore = trainedSyntheticNetwork.evaluate(xAdvIFGSMTargeted, yAdvTarget, verbose=2)
    print("Targeted IFGSM attack score:", ifgsmTargetScore[1])
    DataManager.SaveAdversarialData("IFGSMTargeted"+saveTag, xClean, yClean, xAdvIFGSMTargeted)

    #MIM targeted 
    xAdvMIMTargeted = AttackWrappersWhiteBox.MIMTargeted(sess, trainedSyntheticNetwork, xClean, yAdvTarget, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, decayFactorMIM, clipMin, clipMax)
    xAdvMIMTargetedScore = trainedSyntheticNetwork.evaluate(xAdvMIMTargeted, yAdvTarget, verbose=2)
    print("Targeted MIM attack score:", xAdvMIMTargetedScore[1])
    DataManager.SaveAdversarialData("MIMTargeted"+saveTag, xClean, yClean, xAdvMIMTargeted)

    #PGD targeted 
    xAdvPGDTargeted = AttackWrappersWhiteBox.PGDTargeted(sess, trainedSyntheticNetwork, xClean, yAdvTarget, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, clipMin, clipMax)
    xAdvPGDTargetScore = trainedSyntheticNetwork.evaluate(xAdvPGDTargeted, yAdvTarget, verbose=2)
    print("Targeted PGD attack score:", xAdvPGDTargetScore[1])
    DataManager.SaveAdversarialData("PGDTargeted"+saveTag, xClean, yClean, xAdvPGDTargeted)

    #FGSM untargeted 
    xAdvFGSMNOTarget = AttackWrappersWhiteBox.FGSMNOTarget(sess, trainedSyntheticNetwork,  xClean, yClean, epsFGSM, clipMin, clipMax)
    fgsmNOTargetScore = trainedSyntheticNetwork.evaluate(xAdvFGSMNOTarget, yClean)
    defenseAccuracyFGSMNOTargeted = 1 - fgsmNOTargetScore[1]
    print("Untargeted FGSM attack score:", defenseAccuracyFGSMNOTargeted)
    DataManager.SaveAdversarialData("FGSMNOTargeted"+saveTag, xClean, yClean, xAdvFGSMNOTarget)

    #BIM untargeted 
    xAdvIFGSMNOTargeted = AttackWrappersWhiteBox.PGDNotTargeted(sess, trainedSyntheticNetwork, xClean, yClean, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, 0, clipMin, clipMax)
    ifgsmNOTargetScore = trainedSyntheticNetwork.evaluate(xAdvIFGSMNOTargeted, yClean, verbose=2)
    defenseAccuracyIFGSMNOTargeted = 1 - ifgsmNOTargetScore[1]
    print("Untargeted IFGSM attack score:", defenseAccuracyIFGSMNOTargeted)
    DataManager.SaveAdversarialData("IFGSMNOTargeted"+saveTag, xClean, yClean, xAdvIFGSMNOTargeted)

    #MIM untargeted
    xAdvMIMNOTarget = AttackWrappersWhiteBox.MIMNotTargeted(sess, trainedSyntheticNetwork,  xClean, yClean, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, decayFactorMIM, clipMin, clipMax)
    xAdvMIMNOTargetedScore = trainedSyntheticNetwork.evaluate(xAdvMIMNOTarget, yClean, verbose=2)
    defenseAccuracyMIMNOTargeted = 1 - xAdvMIMNOTargetedScore[1]
    print("Untargeted MIM attack score:", defenseAccuracyMIMNOTargeted)
    DataManager.SaveAdversarialData("MIMNOTargeted"+saveTag, xClean, yClean, xAdvMIMNOTarget)

    #PGD untargeted 
    xAdvPGDNOTargeted = AttackWrappersWhiteBox.PGDNotTargeted(sess, trainedSyntheticNetwork, xClean, yClean, maxIterationsBIM, epsForBIM, maxChangeEpsBIM, rStartPGD, clipMin, clipMax)
    xAdvPGDNOTargetedScore = trainedSyntheticNetwork.evaluate(xAdvPGDNOTargeted, yClean, verbose=2)
    defenseAccuracyPGDNOTargeted = 1 - xAdvPGDNOTargetedScore[1]
    print("Untargeted PGD attack score:", defenseAccuracyPGDNOTargeted)
    DataManager.SaveAdversarialData("PGDNOTargeted"+saveTag, xClean, yClean, xAdvPGDNOTargeted)

    #Carlini L2 Targeted 
    xAdvCarliniTargeted = AttackWrappersWhiteBox.CarliniL2Targeted(sess, attackBatchSizeCarlini, maxIterationsCarlini, trainedSyntheticNetwork, xClean, yAdvTarget, clipMin, clipMax)
    xAdvCarliniTargetedScore = trainedSyntheticNetwork.evaluate(xAdvCarliniTargeted, yAdvTarget, verbose=2)
    print("Targeted Carlini attack score:", xAdvCarliniTargetedScore[1])
    DataManager.SaveAdversarialData("CarliniL2Targeted"+saveTag, xClean, yClean, xAdvCarliniTargeted)

    #Carlini L2 Untargeted 
    xAdvCarliniNOTargeted = AttackWrappersWhiteBox.CarliniL2NOTarget(sess, attackBatchSizeCarlini, maxIterationsCarlini, trainedSyntheticNetwork, xClean, yClean, clipMin, clipMax)
    xAdvCarliniNOTargetedScore = trainedSyntheticNetwork.evaluate(xAdvCarliniNOTargeted, yClean, verbose=2)
    defenseAccuracyCarliniNOTargeted = 1 - xAdvCarliniNOTargetedScore[1]
    print("Untargeted Carlini attack score:", defenseAccuracyCarliniNOTargeted)
    DataManager.SaveAdversarialData("CarliniL2Untargeted"+saveTag, xClean, yClean, xAdvCarliniNOTargeted)

    #EAD Targeted 
    #We use the Carlini attack parameters again
    confidence = 0.0
    learningRate = 1e-2
    binarySearchSteps = 9
    initialConstant = 1e-3
    xAdvEADTargeted = AttackWrappersWhiteBox.ElasticNetTargeted(sess, trainedSyntheticNetwork, xClean, yAdvTarget, betaEAD, attackBatchSizeCarlini, confidence, learningRate, binarySearchSteps, maxIterationsCarlini, initialConstant, clipMin, clipMax)
    xAdvEADTargetedScore = trainedSyntheticNetwork.evaluate(xAdvEADTargeted, yAdvTarget, verbose=2)
    print("Targeted EAD attack score:", xAdvEADTargetedScore)
    DataManager.SaveAdversarialData("EADTargeted"+saveTag, xClean, yClean, xAdvEADTargeted)

    #EAD Untargeted 
    xAdvEADNOTarget = AttackWrappersWhiteBox.ElasticNetNOTargeted(sess, trainedSyntheticNetwork,  xClean, yClean, betaEAD, attackBatchSizeCarlini, confidence, learningRate, binarySearchSteps, maxIterationsCarlini, initialConstant, clipMin, clipMax)
    xAdvEADNOTargetScore = trainedSyntheticNetwork.evaluate(xAdvEADNOTarget, yClean, verbose=2)
    defenseAccuracyEADNOTargeted = 1 - xAdvEADNOTargetScore[1]
    print("Untargeted EAD attack score:", defenseAccuracyEADNOTargeted)
    DataManager.SaveAdversarialData("EADUntargeted"+saveTag, xClean, yClean, xAdvEADNOTarget)

    #Check the clean accuracy 
    cleanScore = trainedSyntheticNetwork.evaluate(xTest, yTest, verbose=2)
    print("The clean score is:", cleanScore[1])
    os.chdir("..") #move up one directory to return to original directory 

#Load the adversarially generated samples and run the attack on an oracle model 
#Oracle model should be of  class type multi-model OR be have evaluate and evaluateAdversarialAttackSuccessRate methods  
def RunPureBlackBoxAttack(advSampleDir, oracleModel, saveTagAdvSamples, saveTagResults):
    attackListTargeted = ['FGSMTargeted', 'IFGSMTargeted', 'MIMTargeted', 'PGDTargeted', 'CarliniL2Targeted', 'EADTargeted']
    attackListNOTarget =['FGSMNOTargeted', 'IFGSMNOTargeted', 'MIMNOTargeted', 'PGDNOTargeted', 'CarliniL2Untargeted', 'EADUntargeted']

    #Create place to save all files
    today = date.today()
    dateString = today.strftime("%B"+"-"+"%d"+"-"+"%Y, ") #Get the year, month, day
    experimentDateAndName = dateString + saveTagResults #Name of experiment with data 
    resultsTextFile = open(experimentDateAndName+", Results.txt","a+")

    #Go through the targeted attacks 
    for i in range(0, len(attackListTargeted)):
        currentAttackName = attackListTargeted[i] #Get the name of the attack 
        np_load_old = numpy.load
        numpy.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        if currentAttackName == "FGSMTargeted": #We only saved the target y labels for this attack but it is fine since all targeted attacks use the same set of labels 
            xClean, yClean, xAdv, yAdv = DataManager.LoadAdversarialData(advSampleDir, currentAttackName+saveTagAdvSamples, targeted=True)
        else:
            xClean, yClean, xAdv = DataManager.LoadAdversarialData(advSampleDir, currentAttackName+saveTagAdvSamples, targeted=False)
        numpy.load = np_load_old
        attackScore = oracleModel.evaluate(xAdv, yAdv) 
        defenseAccuracy = 1 - attackScore #this targeted attack so sample must have target label or else attack fails
        print(str(currentAttackName)+" defense accuracy:", defenseAccuracy)
        resultsTextFile.write(str(currentAttackName)+" defense accuracy:"+str(defenseAccuracy)+"\n")

    #Go through the targeted attacks 
    for i in range(0, len(attackListNOTarget)):
        currentAttackName = attackListNOTarget[i] #Get the name of the attack 
        np_load_old = numpy.load
        numpy.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        xClean, yClean, xAdv = DataManager.LoadAdversarialData(advSampleDir, currentAttackName+saveTagAdvSamples, targeted=False)
        numpy.load = np_load_old
        attackScore = oracleModel.evaluateAdversarialAttackSuccessRate(xAdv, yClean)
        defenseAccuracy = 1 - attackScore #this untargeted attack so must have correct label OR adversarial label for attack to fail
        print(str(currentAttackName)+" defense accuracy:", defenseAccuracy)
        resultsTextFile.write(str(currentAttackName)+" defense accuracy:"+str(defenseAccuracy)+"\n")

