import tensorflow as tf
from tensorflow.keras.layers import Layer, BatchNormalization, Dropout, Lambda, Input, Dense, Conv2D, Flatten, Activation, Concatenate, concatenate, GaussianNoise
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras import backend as K
import scipy.linalg
from tensorflow.python.keras.layers import Lambda
import pickle
import numpy as np

tf.set_random_seed(1) 

#import DataManager
#sess =  K.get_session()

class DataAugmenter(Layer):
    """Shifts and scales input
    Only active at training time since it is a regularization layer.
    # Arguments
        attenuation: how much to attenuate the input
    # Input shape
        Arbitrary.
    # Output shape
        Same as the input shape.
    """

    def __init__(self,  batch_size, **kwargs):
        super(DataAugmenter, self).__init__(**kwargs)
        self.supports_masking = True
        self.batch_size = batch_size




    def call(self, inputs, training=None):
        def augmented():
            


            angles = (15*(2*np.random.rand(self.batch_size)-1))*np.pi/180
            shifts = 4*(2*np.random.rand(self.batch_size, 2)-1) 

            inputs_shifted = tf.contrib.image.translate(inputs, shifts)
            inputs_shifted_rotated = tf.contrib.image.rotate(inputs_shifted,angles)
            
            random_number = tf.random_uniform([self.batch_size])   
            inputs_shifted_rotated_flipped = tf.where(random_number<0.5, tf.image.flip_left_right(inputs_shifted_rotated), inputs_shifted_rotated)
                
            return inputs_shifted_rotated_flipped

           
        
        return K.in_train_phase(augmented, inputs, training=training)

    def get_config(self):
        config = {}
        config['batch_size'] = self.batch_size
        base_config = super(DataAugmenter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Grayscaler(Layer):
    """Converts input to grayscale
    Only active at training time since it is a regularization layer.
    # Arguments
        attenuation: how much to attenuate the input
    # Input shape
        Arbitrary.
    # Output shape
        Same as the input shape.
    """

    def __init__(self,  **kwargs):
        super(Grayscaler, self).__init__(**kwargs)
        self.supports_masking = True



    def call(self, inputs, training=None):
        def augmented():            
            return tf.image.rgb_to_grayscale(inputs)
                        
        return K.in_train_phase(augmented, augmented, training=training)
    
    
    

    def get_config(self):
        config = {}
        base_config = super(Grayscaler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Clipper(Layer):
    """clips input to lie wihin valid pixel range
    Only active at training time since it is a regularization layer.
    # Arguments
        attenuation: how much to attenuate the input
    # Input shape
        Arbitrary.
    # Output shape
        Same as the input shape.
    """

    def __init__(self,  **kwargs):
        super(Clipper, self).__init__(**kwargs)
        self.supports_masking = True


    def call(self, inputs, training=None):
        def augmented():            
            return tf.clip_by_value(inputs,-0.5,0.5)
                        
        return K.in_train_phase(augmented, augmented, training=training)

    

    def get_config(self):
        config = {}
        base_config = super(Clipper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ClassBlender(Layer):
    """Only active at training time since it is a regularization layer.
    # Arguments
        attenuation: how much to attenuate the input
    # Input shape
        Arbitrary.
    # Output shape
        Same as the input shape.
    """

    def __init__(self,  attenuation, batch_size, **kwargs):
        super(ClassBlender, self).__init__(**kwargs)
        self.supports_masking = True
        self.attenuation = attenuation
        self.batch_size = batch_size


    def call(self, inputs, training=None):
        def blended():
    
            inputs_permuted = tf.random_shuffle(inputs)
            angles = (180*(2*np.random.rand(self.batch_size)-1))*np.pi/180
            shifts = 4*(2*np.random.rand(self.batch_size, 2)-1) 
            inputs_permuted_translated = tf.contrib.image.translate(inputs_permuted, shifts)
            inputs_permuted_translated_rotated = tf.contrib.image.rotate(inputs_permuted_translated,angles)         
            inputs_adjusted = inputs_permuted_translated_rotated 
         
            inputs_adjusted = tf.clip_by_value(inputs_adjusted,-0.5,0.5)
            
            
            return (1.0-self.attenuation)*inputs + self.attenuation*inputs_adjusted
            
        
        return K.in_train_phase(blended, inputs, training=training)

    def get_config(self):
        config = {'attenuation': self.attenuation, 'batch_size':self.batch_size}
        base_config = super(ClassBlender, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class WeightsSaver(Callback):
    def __init__(self, N):
        self.N = N
        self.epoch = 0
        
    def specifyFilePath(self, path):
        self.full_path = path #full path to file, including file name
        
    def on_epoch_end(self, epoch, logs={}):
       if self.epoch % self.N == 0:
            print("SAVING WEIGHTS")
            w= self.model.get_weights()
            pklfile= self.full_path + '_' + str(self.epoch) + '.pkl'
            fpkl= open(pklfile, 'wb')        
            pickle.dump(w, fpkl)
            fpkl.close()
       self.epoch += 1


class Model():
    
    def __init__(self, params_dict):

        self.params_dict = params_dict
        self.input = Input(shape=self.params_dict['inp_shape'], name='input') 
 
        self.TRAIN_FLAG=0
        #self.encodeData()
        #map categorical class labels (numbers) to encoded (e.g., one hot or ECOC) vectors
    def encodeData(self, X, Y):
 
        Y_test = np.zeros((X.shape[0], self.params_dict['M'].shape[1]))
        for k in np.arange(self.params_dict['M'].shape[1]):
            Y_test[:,k] = self.params_dict['M'][Y, k]

        return X, Y_test

    def resize(self, x):  
            if self.current_shape[0] == None:
                #print("resize")
                x_input_shape = np.array(self.current_shape[1:4], dtype=np.int)
                x = tf.reshape(x, (-1, x_input_shape[0], x_input_shape[1], x_input_shape[2])) #x.shape = None, 32, 32, 3
            return x

    #define the neural network
    def defineModel(self):


        outputs=[]
        self.penultimate = []
        self.penultimate2 = []
        
        n = int(self.params_dict['M'].shape[1]/self.params_dict['num_chunks'])
        for k in np.arange(0,self.params_dict['num_chunks']):
            
            x = self.input 
            
            if self.params_dict['inp_shape'][2]>1:
                x_gs = Grayscaler()(x)
            else:
                x_gs = x
           
            if (self.TRAIN_FLAG==1):
                x = GaussianNoise(self.params_dict['noise_stddev'], input_shape=self.params_dict['inp_shape'])(x)
                x_gs = GaussianNoise(self.params_dict['noise_stddev'], input_shape=self.params_dict['inp_shape'])(x_gs)

                if self.params_dict['DATA_AUGMENTATION_FLAG']>0:
 
                  
                    self.current_shape = x.get_shape().as_list()
                    x = DataAugmenter(self.params_dict['batch_size'])(x) 
                    x = Lambda(self.resize)(x) #have to resize the shape as (None, 32, 32, 3)
                                            #otherwise the translate function in the DataAugmenter returns output with 'None' shape and later causes error contructing the network


                    self.current_shape = x_gs.get_shape().as_list()
                    x_gs = DataAugmenter(self.params_dict['batch_size'])(x_gs)
                    x_gs = Lambda(self.resize)(x_gs)
                   

                x = ClassBlender(self.params_dict['blend_factor'], self.params_dict['batch_size'])(x)  
                x_gs = ClassBlender(self.params_dict['blend_factor'], self.params_dict['batch_size'])(x_gs)  
            
             
            x = Clipper()(x)
            x_gs = Clipper()(x_gs)
                                    
            for rep in np.arange(self.params_dict['model_rep']):
                x = Conv2D(self.params_dict['num_filters_ens'][0], (5,5), activation='elu', padding='same')(x)          
                if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                    x = BatchNormalization()(x)
            

            x = Conv2D(self.params_dict['num_filters_ens'][0], (3,3), strides=(2,2), activation='elu', padding='same')(x)
            if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                x = BatchNormalization()(x)


            for rep in np.arange(self.params_dict['model_rep']):
                x = Conv2D(self.params_dict['num_filters_ens'][1], (3, 3), activation='elu', padding='same')(x)
                if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                    x = BatchNormalization()(x)
            
            x = Conv2D(self.params_dict['num_filters_ens'][1], (3,3), strides=(2,2), activation='elu', padding='same')(x)  
            if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                x = BatchNormalization()(x)
            
            for rep in np.arange(self.params_dict['model_rep']):
                x = Conv2D(self.params_dict['num_filters_ens'][2], (3, 3), activation='elu', padding='same')(x)
                if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                    x = BatchNormalization()(x)
            
            
            x = Conv2D(self.params_dict['num_filters_ens'][2], (3,3), strides=(2,2), activation='elu', padding='same')(x)  

                        
            pens = []
            out=[]
            for k2 in np.arange(n):
                x0 = Conv2D(self.params_dict['num_filters_ens_2'], (5, 5), strides=(2,2), activation='elu', padding='same')(x_gs)
                x0 = Conv2D(self.params_dict['num_filters_ens_2'], (3, 3), strides=(2,2), activation='elu', padding='same')(x0)
                x0 = Conv2D(self.params_dict['num_filters_ens_2'], (3, 3), strides=(2,2), activation='elu', padding='same')(x0)

                x_= Concatenate()([x0, x]) 
            
            
                x_ = Conv2D(self.params_dict['num_filters_ens_2'], (2, 2), activation='elu', padding='same')(x_)                    
                    
                x_ = Conv2D(self.params_dict['num_filters_ens_2'], (2, 2), activation='elu', padding='same')(x_)

                x_ = Flatten()(x_)

                x_ = Dense(16, activation='elu')(x_) 
                x_ = Dense(8, activation='elu')(x_) 
                x_ = Dense(4, activation='elu')(x_) 
                x0 = Dense(2, activation='linear')(x_) 

                pens += [x0]                

                x1 = Dense(1, activation='linear', name='w_'+str(k)+'_'+str(k2)+'_'+self.params_dict['name'], kernel_regularizer=regularizers.l2(0.0))(x0) 
                out += [x1]
                
            self.penultimate += [pens]
            
            if len(pens) > 1:
                self.penultimate2 += [Concatenate()(pens)]
            else:
                self.penultimate2 += pens

            if len(out)>1:
                outputs += [Concatenate()(out)]
            else:
                outputs += out


        self.model = KerasModel(inputs=self.input, outputs=outputs)
        print(self.model.summary())

        return outputs



    def defineLoss(self, idx):    
        def hinge_loss(y_true, y_pred):
            loss = tf.reduce_mean(tf.maximum(1.0-y_true*y_pred, 0))
            return loss   
        
        return hinge_loss
        
    
    
    def defineMetric(self):
        def hinge_pred(y_true, y_pred):
            corr = tf.to_float((y_pred*y_true)>0)
            return tf.reduce_mean(corr)
        return [hinge_pred]
          

    def saveModel(self):
        w= self.model.get_weights()
        pklfile= self.params_dict['model_path'] + self.params_dict['name'] + '_final.pkl'
        fpkl= open(pklfile, 'wb')        
        pickle.dump(w, fpkl)
        fpkl.close()
        self.model.save(self.params_dict['model_path'] + self.params_dict['name'] + '_final.h5')

        
    def trainModel(self, X_train, X_test, Y_train, Y_test):
        opt = Adam(lr=self.params_dict['lr'])
        
        self.model.compile(optimizer=opt, loss=[self.defineLoss(k) for k in np.arange(self.params_dict['num_chunks'])], metrics=self.defineMetric())
        WS = WeightsSaver(self.params_dict['weight_save_freq'])
        WS.specifyFilePath(self.params_dict['model_path'] + self.params_dict['name'])
        
        X_train, Y_train = self.encodeData(X_train, Y_train)
        X_test, Y_test = self.encodeData(X_test, Y_test)

        Y_train_list=[]
        Y_test_list=[]

        start = 0
        
        for k in np.arange(self.params_dict['num_chunks']):
            end = start + int(self.params_dict['M'].shape[1]/self.params_dict['num_chunks'])
            Y_train_list += [Y_train[:,start:end]]
            Y_test_list += [Y_test[:,start:end]]
            start=end
            

        self.model.fit(X_train, Y_train_list,
                            epochs=self.params_dict['epochs'], 
                            batch_size=self.params_dict['batch_size'],
                            shuffle=True,
                            validation_data=[X_test, Y_test_list],
                            callbacks=[WS])
        
        self.saveModel()

    def outputDecoder(self, x):
        
        mat1 = tf.matmul(x, self.params_dict['M'], transpose_b=True)
        mat1 = tf.log(tf.maximum(mat1, 0)+1e-6) #floor negative values
        return mat1 

    def defineFullModel(self):
        self.TRAIN_FLAG=0
        outputs = self.defineModel()
        
        if len(outputs)>1:
            self.raw_output = Concatenate()(outputs)
        else: #if only a single chunk
            self.raw_output = outputs[0]
            
        #pass output logits through activation
        for idx,o in enumerate(outputs):
            outputs[idx] = Lambda(self.params_dict['output_activation'])(o)
            
        if len(outputs)>1:
            x = Concatenate()(outputs)
        else: #if only a single chunk
            x = outputs[0]
        x = Lambda(self.outputDecoder)(x) #logits
        x = Activation('softmax')(x) #return probs
        
        if self.params_dict['base_model'] == None:
            self.model_full = KerasModel(inputs=self.input, outputs=x)
        else:
            self.model_full = KerasModel(inputs=self.params_dict['base_model'].input, outputs=x)

    def loadFullModel(self, pklfile):
        f= open(pklfile, 'rb')
        weigh= pickle.load(f);  
        f.close();
        self.defineFullModel()
        self.model_full.set_weights(weigh)

        #path = 'tanh_32_diverse_cifar10_final.h5'
        #self.defineFullModel()
        ##self.model_full.set_weights(weigh)
        #self.model.load_weights(path)

    def predict(self, X):
        return self.model_full(X)

