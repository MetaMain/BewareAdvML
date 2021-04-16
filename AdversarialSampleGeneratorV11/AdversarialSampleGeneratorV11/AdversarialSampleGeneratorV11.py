import tensorflow as tf
from tensorflow import keras
import os 
import winsound
import DataManager
import DefaultMethodsComDefend
import DefaultMethodsVanilla
import DefaultMethodsBuzz
import DefaultMethodsBarrage
import DefaultMethodsADP
import DebugMethods
import DefaultMethodsOdds
import DefaultMethodsBuzzFashion
import DefaultMethodsECOC

#Make sure the correct GPU device is used with tensor flow 
def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0" #Make it so only the correct GPU is seen by the program 
    #Configuration for the GPU 
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)
    with tf.device('/job:localhost/replica:0/task:0/device:GPU:0'):
        #DefaultMethodsBarrage.PureBlackBoxRunAttacksOnAllTransformationsCIFAR10()
        #DefaultMethodsBarrage.PureBlackBoxRunAttacksOnAllTransformationsFashionMNIST()
        #DefaultMethodsOdds.PapernotAttackFashionMNIST(sess)
        #DefaultMethodsADP.RunAllCIFAR10Attacks()
        #DebugMethods.TestModelFMNIST()
        #DebugMethods.TestPureBlackBoxAttackCIFAR10()
        #DefaultMethodsOdds.PureBlackBoxRunAttacksOnAllTransformationsCIFAR10()
        #DefaultMethodsOdds.PureBlackBoxRunAttacksOnAllTransformationsFashionMNIST()
        #DefaultMethodsBuzzFashion.RunAllFmnistBUZZTwoAttacks()
        #DefaultMethodsBuzzFashion.RunAllFmnistBUZZEightAttacks()
        #DefaultMethodsComDefend.PureBlackBoxRunAttacksOnAllTransformationsFashionMNIST()
        #DefaultMethodsComDefend.PureBlackBoxRunAttacksOnAllTransformationsCIFAR10()
        #DefaultMethodsADP.PureBlackBoxRunAttacksCIFAR10()
        #DefaultMethodsADP.PureBlackBoxRunAttacksFashionMNIST()
        #DefaultMethodsBuzz.PureBlackBoxRunAttacksBUZZ2CIFAR10()
        #DefaultMethodsBuzz.PureBlackBoxRunAttacksBUZZ8CIFAR10()
        #DefaultMethodsBuzzFashion.PureBlackBoxRunAttacksBUZZ8FashionMNIST()
        #DefaultMethodsBuzzFashion.PureBlackBoxRunAttacksBUZZ2FashionMNIST()
        #DefaultMethodsECOC.RunAllCIFAR10Attacks()
        DefaultMethodsECOC.RunAllFashionMNISTAttacks()

if __name__ == '__main__':
    main()

