import numpy

class DefenseMultiModel():
    #Default constructor 
    def __init__(self ,secretModelList, classNum, threshold):
        self.SecretModelList=secretModelList
        self.ClassNum=classNum
        self.ModelNum=len(self.SecretModelList)
        self.Threshold=threshold

    #Majority voting AND thresholding 
    def predict(self, xData):
        sampleSize=xData.shape[0]
        modelVotes=numpy.zeros((self.ModelNum,sampleSize,self.ClassNum)) 

        #Get the votes for each of the networks 
        for i in range(0,self.ModelNum):
            modelVotes[i,:,:]=self.SecretModelList[i].predict(xData)
        
        #Now do the voting 
        finalVotes=numpy.zeros((sampleSize,self.ClassNum+1)) #The 11th class is the noise class
        for i in range(0, sampleSize):
            currentTally=numpy.zeros((self.ClassNum,))
            for j in range(0, self.ModelNum):
                currentVote=modelVotes[j,i,:].argmax(axis=0)
                currentTally[currentVote]=currentTally[currentVote]+1
            if (currentTally[currentTally.argmax(axis=0)]>=self.Threshold): #Make sure it is above the threshold 
                finalVotes[i,currentTally.argmax(axis=0)]=1.0
            else: #Make it the last "noise" class
                finalVotes[i,self.ClassNum]=1.0
        return finalVotes

    def evaluate(self, xTest, yTest):
        accuracy=0
        sampleSize=xTest.shape[0]
        multiModelOutput=self.predict(xTest)
        for i in range(0, sampleSize):
            if(multiModelOutput[i].argmax(axis=0)==yTest[i].argmax(axis=0)):
                accuracy=accuracy+1
        accuracy=accuracy/sampleSize
        return accuracy

    #the network is fooled if we don't have a noise class label AND it gets the wrong label 
    #Returns attack success rate 
    def evaluateAdversarialAttackSuccessRate(self, xAdv, yClean):
        sampleNum=xAdv.shape[0]
        yPred=self.predict(xAdv)
        advAcc=0
        for i in range(0, sampleNum):
            #The attack wins only if we don't correctly label the sample AND the sample isn't given the nosie class label
            if yPred[i].argmax(axis=0) != self.ClassNum and yPred[i].argmax(axis=0) != yClean[i].argmax(axis=0): #The last class is the noise class
                advAcc=advAcc+1
        advAcc=advAcc/sampleNum
        return advAcc
