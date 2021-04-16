import numpy
import tensorflow 
from tensorflow import keras
Model = keras.models.Model
Dense = keras.layers.Dense
Dropout=keras.layers.Dropout
Activation = keras.layers.Activation
Flatten = keras.layers.Flatten
BatchNormalization= keras.layers.BatchNormalization
Conv2D = keras.layers.Conv2D
Dropout=keras.layers.Dropout
MaxPooling2D=keras.layers.MaxPooling2D
Input=keras.layers.Input
regularizers=keras.regularizers

#VGG16 with possible B, A, and G (plus all the other things, resize, zeropad, etc)
def GenerateBasePrivateVgg16Model(inputShape, input_img, num_classes, resizeValue=None, zeroPadValue=None , aMatrix=None, bMatrix=None, pertValue=None):
    weight_decay = 0.0005
    img_rows=inputShape[0]
    img_cols=inputShape[1]
    colorChannelNum=inputShape[2]

    #Resize Layer 
    #First check app constants, if we need to resize change from original value, otherwise keep it same as input  
    if resizeValue is None:
        resizeValueX=img_rows
        resizeValueY=img_cols
    else:
        resizeValueX=resizeValue
        resizeValueY=resizeValue
    
    #Now time to configure the zero padding layer 
    if zeroPadValue is None:
        zeroPadValueX=0
        zeroPadValueY=0
    else:
        zeroPadValueX=zeroPadValue
        zeroPadValueY=zeroPadValue

    #Make sure the image does not drop below the original size by applying zero padding if necessary 
    if resizeValueX<img_rows: 
        extraPad=int(img_rows-resizeValueX)
        zeroPadValueX=zeroPadValueX+extraPad

    if resizeValueY<img_cols: 
        extraPad=int(img_cols-resizeValueY)
        zeroPadValueY=zeroPadValueY+extraPad

    #Configure the B matrix layer 
    #currentImageSizeX=resizeValueX+2*zeroPadValueX #The zero pad is applied left and right / top and bottom 
    #currentImageSizeY=resizeValueY+2*zeroPadValueY

    totalCurrentPixels=int(img_rows*img_cols)
    if bMatrix is None:
        bMatrix=numpy.zeros((totalCurrentPixels,)) #If we don't have a bMatrix just make it a matrix with all 0s
        #print("Insanity check: You should not see this message for B")
    if bMatrix.shape[0] != totalCurrentPixels:
        raise ValueError("The B matrix dimensions are not correct for the resized image.")

    if aMatrix is None: #Use the identity matrix if no other matrix is provided 
        tempWeights=numpy.zeros((totalCurrentPixels, totalCurrentPixels))
        for i in range(0, totalCurrentPixels):
            tempIndex=i
            tempWeights[i,tempIndex]=1.0
    else: #There has been an A matrix provided so use it.
        print("Sanity check, A matrix has been set.")
        tempWeights=aMatrix

    #Time to put in layers
#    k1=keras.layers.Permute((3,1,2))(input_img)
#    k2=keras.layers.Reshape((colorChannelNum, totalCurrentPixels))(k1)
#    k3=Dense(totalCurrentPixels, activation='linear', weights=[tempWeights, bMatrix], trainable=False)(k2)  #name='BMatrixLayer'
#    k4=keras.layers.Reshape((colorChannelNum, img_rows, img_cols))(k3)
#    k5=keras.layers.Permute((2,3,1))(k4) 
#    resizeLayer= keras.layers.Lambda(lambda image: tensorflow.image.resize_images(image,(resizeValueX, resizeValueY), method = tensorflow.image.ResizeMethod.BICUBIC, align_corners = True, preserve_aspect_ratio = True))(k5)
#    padLayer =  tensorflow.keras.layers.ZeroPadding2D((zeroPadValueX, zeroPadValueY))(resizeLayer)#Zeropad Layer 

    resizeLayer= keras.layers.Lambda(lambda image: tensorflow.image.resize_images(image,(resizeValueX, resizeValueY), method = tensorflow.image.ResizeMethod.BICUBIC, align_corners = True, preserve_aspect_ratio = True))(input_img)

    #These are the layers used for classification (standard VGG-16)
    #z_1=Conv2D(64, (3, 3), padding='same', input_shape=inputShape,kernel_regularizer=regularizers.l2(weight_decay), name='z_1', activation='relu')(padLayer)
   # z_1=Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(padLayer) # name='z_1'
    z_1=Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(resizeLayer) # name='z_1'

    z_1B=BatchNormalization()(z_1)
    z_1D=Dropout(0.3)(z_1B)

    z_2=Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(z_1D) # name='z_1'
    z_2B=BatchNormalization()(z_2)
    z_2MP=MaxPooling2D(pool_size=(2, 2))(z_2B)

    z_3=Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(z_2MP) # name='z_3'
    z_3B=BatchNormalization()(z_3)
    z_3D=Dropout(0.4)(z_3B)

    z_4=Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(z_3D) #name='z_4'
    z_4B=BatchNormalization()(z_4)
    z_4MP=MaxPooling2D(pool_size=(2, 2))(z_4B)

    z_5=Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(z_4MP)
    z_5B=BatchNormalization()(z_5)
    z_5D=Dropout(0.4)(z_5B)

    z_6=Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(z_5D)
    z_6B=BatchNormalization()(z_6)
    z_6D=Dropout(0.4)(z_6B)

    z_7=Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(z_6D)
    z_7B=BatchNormalization()(z_7)

    z_7MP=MaxPooling2D(pool_size=(2, 2))(z_7B)

    z_8=Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(z_7MP)
    z_8B=BatchNormalization()(z_8)
    z_8D=Dropout(0.4)(z_8B)

    z_9=Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(z_8D)
    z_9B=BatchNormalization()(z_9)
    z_9D=Dropout(0.4)(z_9B)

    z_10=Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay),  activation='relu')(z_9D)
    z_10B=BatchNormalization()(z_10)
    z_10MP=MaxPooling2D(pool_size=(2, 2))(z_10B)

    z_11=Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay),  activation='relu')(z_10MP)
    z_11B=BatchNormalization()(z_11)
    z_11D=Dropout(0.4)(z_11B)

    z_12=Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay),activation='relu')(z_11D)
    z_12B=BatchNormalization()(z_12)
    z_12D=Dropout(0.4)(z_12B)

    z_13=Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay),  activation='relu')(z_12D)
    z_13B=BatchNormalization()(z_13)

    z_13MP=MaxPooling2D(pool_size=(2, 2))(z_13B)
    z_13D=Dropout(0.5)(z_13MP) #Double check that drop out should really go here 

    flat=Flatten()(z_13D)

    #We don't have a secret transformation layer inbetween the CL and FF
    if pertValue is None: 
        z_14=Dense(512,kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(flat)
        z_14B=BatchNormalization()(z_14)
        z_15=Dense(512,kernel_regularizer=regularizers.l2(weight_decay), activation='relu')(z_14B)
        z_15B=BatchNormalization()(z_15)
        z_15D=Dropout(0.5)(z_15B)
        z_16=Dense(num_classes,kernel_regularizer=regularizers.l2(weight_decay), activation='softmax')(z_15D)
        modelFull = Model(inputs=input_img, outputs=z_16)
        return z_16
    else: #There is a g-layer between the CL and FF
        #Define a dummy model (there may be a better way to do this)
        tempModel=Model(inputs=input_img, outputs=flat)
        templastLayerIndex=len(tempModel.layers)
        flatDim=tempModel.layers[templastLayerIndex-1].output_shape[1]
        #Component secret transformation g applied here 
        secretInputSize1D=512 #right now this is hardcoded 
        secretA = 2 * pertValue * (numpy.random.rand(flatDim*secretInputSize1D))-pertValue #This is (b-a)*r+a which is equivalent to 2*pertVal*r-pertVal uniform dist~(-pertVal,pertVal)
        secretA=numpy.reshape(secretA,(flatDim, secretInputSize1D))
        secretB = 2* pertValue * (numpy.random.rand(secretInputSize1D))-pertValue
        g_layer=Dense(secretInputSize1D, input_dim=secretInputSize1D, activation='linear', weights=[secretA,secretB], trainable=False)(flat)
        #Now continue with normal VGG16 FF layers 
        z_14=Dense(512,kernel_regularizer=regularizers.l2(weight_decay),  activation='relu')(g_layer)
        z_14B=BatchNormalization()(z_14)
        z_15=Dense(512,kernel_regularizer=regularizers.l2(weight_decay),  activation='relu')(z_14B)
        z_15B=BatchNormalization()(z_15)
        z_15D=Dropout(0.5)(z_15B)
        z_16=Dense(num_classes, kernel_regularizer=regularizers.l2(weight_decay), activation='softmax')(z_15D)
        modelFull = Model(inputs=input_img, outputs=z_16)
        return z_16

