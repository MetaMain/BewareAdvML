import tensorflow as tf
import numpy as np
import time

#from .misc import *

# this library is Python 3 only.

# this library aims to wrap up the ugliness of tensorflow,
# at the same time provide a better interface for NON-STANDARD
# learning experiments(such as GANs, etc.) than Keras.

# a Can is a container. it can contain other Cans.

class Can:
    def __init__(self):
        self.subcans = [] # other cans contained
        self.weights = [] # trainable
        self.biases = []
        self.only_weights = []
        self.variables = [] # should save with the weights, but not trainable
        self.updates = [] # update ops, mainly useful for batch norm
        # well, you decide which one to put into

        self.inference = None

    # by making weight, you create trainable variables
    def make_weight(self,shape,name='W', mean=0., stddev=1e-2, initializer=None):
        mean,stddev = [float(k) for k in [mean,stddev]]
        if initializer is None:
            initial = tf.truncated_normal(shape, mean=mean, stddev=stddev)
        else:
            initial = initializer
        w = tf.Variable(initial,name=name)
        self.weights.append(w)
        self.only_weights.append(w)
        return w

    def make_bias(self,shape,name='b', mean=0.):
        mean = float(mean)
        initial = tf.constant(mean, shape=shape)
        b = tf.Variable(initial,name=name)
        self.weights.append(b)
        self.biases.append(b)
        return b

    # make a variable that is not trainable, by passing in a numpy array
    def make_variable(self,nparray,name='v'):
        v = tf.Variable(nparray,name=name)
        self.variables.append(v)
        return v

    # add an op as update op of this can
    def make_update(self,op):
        self.updates.append(op)
        return op

    # put other cans inside this can, as subcans
    def incan(self,c):
        if hasattr(c,'__iter__'): # if iterable
            self.subcans += list(c)
        else:
            self.subcans += [c]
        # return self

    # another name for incan
    def add(self,c):
        self.incan(c)
        return c

    # if you don't wanna specify the __call__ function manually,
    # you may chain up all the subcans to make one:
    def chain(self):
        def call(i):
            for c in self.subcans:
                i = c(i)
            return i
        self.set_function(call)

    # traverse the tree of all subcans,
    # and extract a flattened list of certain attributes.
    # the attribute itself should be a list, such as 'weights'.
    # f is the transformer function, applied to every entry
    def traverse(self,target='weights',f=lambda x:x):
        l = [f(a) for a in getattr(self,target)] + [c.traverse(target,f) for c in self.subcans]
        # the flatten logic is a little bit dirty
        return list(flatten(l, lambda x:isinstance(x,list)))

    # return weight tensors of current can and it's subcans
    def get_weights(self):
        return self.traverse('weights')

    def get_biases(self):
        return self.traverse('biases')

    def get_only_weights(self): # dont get biases
        return self.traverse('only_weights')

    # return update operations of current can and it's subcans
    def get_updates(self):
        return self.traverse('updates')

    # set __call__ function
    def set_function(self,func):
        self.func = func

    # default __call__
    def __call__(self,i,*args,**kwargs):
        if hasattr(self,'func'):
            return self.func(i,*args,**kwargs)
        else:
            raise NameError('You didnt override __call__(), nor called set_function()/chain()')

    def get_value_of(self,tensors):
        sess = get_session()
        values = sess.run(tensors)
        return values

    def save_weights(self,filename): # save both weights and variables
        with open(filename,'wb') as f:
            # extract all weights in one go:
            w = self.get_value_of(self.get_weights()+self.traverse('variables'))
            print(len(w),'weights (and variables) obtained.')

            # create an array object and put all the arrays into it.
            # otherwise np.asanyarray() within np.savez_compressed()
            # might make stupid mistakes
            arrobj = np.empty([len(w)],dtype='object') # array object
            for i in range(len(w)):
                arrobj[i] = w[i]

            np.savez_compressed(f,w=arrobj)
            print('successfully saved to',filename)
            return True

    def load_weights(self,filename):
        #Add by K to avoid picke=false problems
        np_load_old = np.load
        # modify the default parameters of np.load
        np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
        with open(filename,'rb') as f:
            loaded_w = np.load(f)
            #print('successfully loaded from',filename)
            if hasattr(loaded_w,'items'):
                # compressed npz (newer)
                loaded_w = loaded_w['w']
            else:
                # npy (older)
                pass
            # but we cannot assign all those weights in one go...
            model_w = self.get_weights()+self.traverse('variables')
            if len(loaded_w)!=len(model_w):
                raise NameError('number of weights (variables) from the file({}) differ from the model({}).'.format(len(loaded_w),len(model_w)))
            else:
                assign_ops = [tf.assign(model_w[i],loaded_w[i])
                    for i,_ in enumerate(model_w)]

            sess = get_session()
            sess.run(assign_ops)
            #print(len(loaded_w),'weights assigned.')

            #At end have to make pickle false (added by K)
            np.load = np_load_old
            return True

    def infer(self,i):
        # run function, return value
        if self.inference is None:
            # the inference graph will be created when you infer for the first time
            # 1. create placeholders with same dimensions as the input
            if isinstance(i,list): # if Can accept more than one input
                x = [tf.placeholder(tf.float32,shape=[None for _ in range(len(j.shape))])
                    for j in i]
                print('(infer) input is list.')
            else:
                x = tf.placeholder(tf.float32, shape=[None for _ in range(len(i.shape))])

            # 2. set training state to false, construct the graph
            set_training_state(False)
            y = self.__call__(x)
            set_training_state(True)

            # 3. create the inference function
            def inference(k):
                sess = get_session()
                if isinstance(i,list):
                    res = sess.run([y],feed_dict={x[j]:k[j]
                        for j,_ in enumerate(x)})[0]
                else:
                    res = sess.run([y],feed_dict={x:k})[0]
                return res
            self.inference = inference

        return self.inference(i)

    def summary(self):
        print('-------------------')
        print('Directly Trainable:')
        variables_summary(self.get_weights())
        print('-------------------')
        print('Not Directly Trainable:')
        variables_summary(self.traverse('variables'))
        print('-------------------')

def variables_summary(var_list):
    shapes = [v.get_shape() for v in var_list]
    shape_lists = [s.as_list() for s in shapes]
    shape_lists = list(map(lambda x:''.join(map(lambda x:'{:>5}'.format(x),x)),shape_lists))

    num_elements = [s.num_elements() for s in shapes]
    total_num_of_variables = sum(num_elements)
    names = [v.name for v in var_list]

    print('counting variables...')
    for i in range(len(shapes)):
        print('{:>25}  ->  {:<6} {}'.format(
        shape_lists[i],num_elements[i],names[i]))

    print('{:>25}  ->  {:<6} {}'.format(
    'tensors: '+str(len(shapes)),
    str(total_num_of_variables),
    'variables'))

# you know, MLP
class Dense(Can):
    def __init__(self,num_inputs,num_outputs,bias=True,mean=None, stddev=None, initializer=None):
        super().__init__()
        # for different output unit type, use different noise scales
        if stddev is None:
            stddev = 2. # 2 for ReLU, 1 for linear/tanh
        stddev = np.sqrt(stddev/num_inputs)
        if mean is None: # mean for bias layer
            mean = 0.

        self.W = self.make_weight([num_inputs,num_outputs],stddev=stddev,initializer=initializer)
        self.use_bias = bias
        if bias:
            self.b = self.make_bias([num_outputs],mean=mean)
    def __call__(self,i):
        d = tf.matmul(i,self.W)
        if self.use_bias:
            return d + self.b
        else:
            return d

class LayerNormDense(Dense):
    def __init__(self,*args,**kw):
        super().__init__(*args,**kw)
        nop = args[1]
        self.layernorm = self.add(LayerNorm(nop))

    def __call__(self,i):
        d = tf.matmul(i,self.W)
        d = self.layernorm(d)
        if self.use_bias:
            return d + self.b
        else:
            return d

# apply Dense on last dimension of 1d input. equivalent to 1x1 conv1d.
class TimeDistributedDense(Dense):
    def __call__(self,i):
        s = tf.shape(i)
        b,t,d = s[0],s[1],s[2]
        i = tf.reshape(i,[b*t,d])
        i = super().__call__(i)
        d = tf.shape(i)[1]
        i = tf.reshape(i,[b,t,d])
        return i

# apply Dense on last dimension of Nd input.
class LastDimDense(Dense):
    def __call__(self,i):
        s = tf.shape(i)
        rank = tf.rank(i)
        fore = s[0:rank-1] # foremost dimensions
        last = s[rank-1] # last dimension
        prod = tf.reduce_prod(fore)
        i = tf.reshape(i,[prod,last]) # shape into
        i = super().__call__(i) # call Dense layer
        d = tf.shape(i)[1]
        i = tf.reshape(i,tf.concat([fore,[d]],axis=0)) # shape back
        return i

# expand last dimension by a branching factor. Expect input of shape [Batch Dims]
class Expansion(Can):
    def __init__(self,nip,factor,stddev=1):
        super().__init__()
        self.nip = nip; self.factor = factor; self.nop = nip*factor
        self.W = self.make_weight([nip,factor],stddev=stddev)
        self.b = self.make_bias([nip*factor])

    def __call__(self,i):
        # input: [Batch Dimin] weight: [Dimin Factor] output: [Batch Dimin Factor]
        result = tf.einsum('bi,if->bif', i, self.W)
        result = tf.reshape(result,[-1,self.nop])
        return result + self.b

# you know, shorthand
class Lambda(Can):
    def __init__(self,f):
        super().__init__()
        self.set_function(f)

# you know, to fit
class Reshape(Can):
    def __init__(self,shape):
        super().__init__()
        self.shape = shape
    def __call__(self,i):
        bs = tf.shape(i)[0] # assume [batch, dims...]
        return tf.reshape(i,[bs]+self.shape)

# you know, nonlinearities
class Act(Can):
    def __init__(self,name,alpha=0.2):
        super().__init__()
        def lrelu(i): # fast impl. with only 1 relu
            negative = tf.nn.relu(-i)
            res = i + negative * (1.0-alpha)
            return res
        def lrelu(i):
            return tf.nn.leaky_relu(i, alpha)

        def selu(x):
            # https://arxiv.org/pdf/1706.02515.pdf
            alpha = 1.6732632423543772848170429916717
            scale = 1.0507009873554804934193349852946
            return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

        def swish(x):
            return x*tf.sigmoid(x) # beta=1 case

        activations = {
            'relu':tf.nn.relu,
            'tanh':tf.tanh,
            'sigmoid':tf.sigmoid,
            'softmax':tf.nn.softmax,
            'elu':tf.nn.elu,
            'lrelu':lrelu,
            'softplus':tf.nn.softplus,
            'selu':selu,
            'swish':swish,
            'relu6':tf.nn.relu6,
        }
        self.set_function(activations[name])

# you know, brain damage
class Drop(Can):
    def __init__(self,prob,switch=None):
        super().__init__()
        self.prob = prob
        self.switch = switch
    def __call__(self,i):
        if self.switch is None: # not using a switch variable (recommended)
            if get_training_state():
                return tf.nn.dropout(i, keep_prob=self.prob)
            else:
                return i
        else:
            # use a switch variable
            # (if the memory is so limited that a separate flow not possible)
            return tf.cond(self.switch,
                lambda:tf.nn.dropout(i,keep_prob=self.prob),
                lambda:i)

# you know, Yann LeCun
class Conv2D(Can):
    # nip and nop: input and output planes
    # k: dimension of kernel, 3 for 3x3, 5 for 5x5
    # rate: atrous conv rate, 1 = not, 2 = skip one
    def __init__(self,nip,nop,k,std=1,usebias=True,rate=1,padding='SAME',stddev=None):
        super().__init__()
        if stddev is None:
            stddev = 2. # 2 for ReLU, 1 for linear/tanh
        if rate>1 and std>1:
            raise('atrous rate can\'t also be greater \
                than one when stride is already greater than one.')

        self.nip,self.nop,self.k,self.std,self.usebias,self.padding,self.rate\
        = nip,nop,k,std,usebias,padding,rate

        self.W = self.make_weight([k,k,nip,nop],stddev=np.sqrt(stddev/(nip*k*k)))
        # self.W = self.make_weight([k,k,nip,nop],stddev=np.sqrt(stddev/(nip*k*k)),
        #     initializer = tf.contrib.framework.convolutional_delta_orthogonal(
        #         gain = stddev, dtype=tf.float32,
        #     )(shape=[k,k,nip,nop])
        # )
        # assume square window
        if usebias==True:
            self.b =self.make_bias([nop])

    def __call__(self,i):
        if self.rate == 1:
            c = tf.nn.conv2d(i,self.W,
                strides=[1, self.std, self.std, 1],
                padding=self.padding)
        else: #dilated conv
            c = tf.nn.atrous_conv2d(i,self.W,
                rate=self.rate,
                padding=self.padding)

        if self.usebias==True:
            return c + self.b
        else:
            return c

class DepthwiseConv2D(Conv2D):
    def __init__(self, nip, nop, stddev=None,*a,**k):
        if stddev is None:
            stddev = 2. # 2 for ReLU, 1 for linear/tanh
        stddev *= nip # scale for depthwise convolution
        super().__init__(nip=nip, nop=nop, stddev=stddev,*a,**k)
    def __call__(self,i):
        c = tf.nn.depthwise_conv2d(i,self.W,
                strides=[1, self.std, self.std, 1],
                padding=self.padding,
                rate=[self.rate, self.rate],
                )

        if self.usebias==True:
            return c + self.b
        else:
            return c

class GroupConv2D(Can):
    def __init__(self,nip,nop,k,num_groups,*a,**kw):
        super().__init__()
        assert nip % num_groups == 0
        assert nop % num_groups == 0
        self.num_groups = num_groups
        self.nipg = nip//num_groups
        self.nopg = nop//num_groups

        self.groups = [Conv2D(self.nipg, self.nopg, k, *a, **kw) for i in range(num_groups)]
        self.incan(self.groups)

    def __call__(self,i):
        out = []
        for idx, conv in enumerate(self.groups):
            inp = i[:, :, :, idx * self.nipg:(idx+1)*self.nipg]
            out.append(conv(inp))
        return tf.concat(out, axis=-1)

class ChannelShuffle(Can): # shuffle the last dimension
    def __init__(self, nip, num_groups):
        super().__init__()
        assert nip % num_groups == 0
        self.nip = nip
        self.num_groups = num_groups
        self.nipg = nip//num_groups

    def __call__(self, i):
        orig_shape = tf.shape(i)
        reshaped = tf.reshape(i, [-1, self.num_groups, self.nipg])
        transposed = tf.transpose(reshaped, perm=[0,2,1])
        output = tf.reshape(reshaped, orig_shape)
        return output

class ShuffleNet(Can):
    def __init__(self, nip, nop, num_groups):
        super().__init__()
        if nip==nop:
            self.std = 1
        elif nip*2 == nop:
            self.std = 2
        else:
            raise Exception('shufflenet unit accept only nip==nop or nip*2==nop.')

        assert nip % num_groups == 0
        self.num_groups = num_groups
        self.nipg = nip//num_groups

        bottleneck_width = nip//4

        self.gc1 = GroupConv2D(nip, bottleneck_width, k=3, num_groups=num_groups, stddev=2)

        self.cs = ChannelShuffle(bottleneck_width, num_groups=num_groups)
        self.dc = DepthwiseConv2D(bottleneck_width,1, k=3, std=self.std)
        self.gc2 = GroupConv2D(bottleneck_width, nip, k=3, num_groups=num_groups, stddev=1)

        self.incan([self.gc1, self.cs, self.dc, self.gc2])

    def __call__(self, i):
        residual = i

        i = self.gc1(i)
        i = Act('relu')(i)
        i = self.cs(i)
        i = self.dc(i)
        i = self.gc2(i)

        if self.std == 1: # dont grow feature map
            out = residual + i
        else: # grow 2x by concatenation
            residual = AvgPool2D(k=3, std=2)(residual)
            out = tf.concat([residual, i], axis=-1)
        return Act('relu')(out)

# upsampling 2d
class Up2D(Can):
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale
    def __call__(self,i):
        scale = self.scale
        s = tf.shape(i) # assume NHWC
        newsize = [s[1]*scale,s[2]*scale]
        return tf.image.resize_nearest_neighbor(i, size=newsize, align_corners=None, name=None)

# assume padding == 'SAME'.
class Deconv2D(Conv2D):
    def __call__(self,i):
        if self.usebias == True: i -= self.b

        s = tf.shape(i)
        return tf.nn.conv2d_transpose(
            i,
            self.W,
            output_shape=[s[0], s[1]*self.std, s[2]*self.std, self.nip],
            strides=[1, self.std, self.std, 1],
        )

# you know, recurrency
class Scanner(Can):
    def __init__(self,f):
        super().__init__()
        self.f = f
    def __call__(self,i,starting_state=None, inferred_state_shape=None):
        # previous state is useful when running online.
        if starting_state is None:
            if inferred_state_shape is None:
                print('(Scanner) cannot/didnot infer state_shape. use shape of input[0] instead. please make sure the input to the Scanner has the same last dimension as the function being scanned.')
                initializer = tf.zeros_like(i[0])
            else:
                print('(Scanner) using inferred_state_shape')
                initializer = tf.zeros(inferred_state_shape, tf.float32)
        else:
            initializer = starting_state
        scanned = tf.scan(self.f,i,initializer=initializer)
        return scanned

# deal with batch input.
class BatchScanner(Scanner):
    def __call__(self, i, **kwargs):
        rank = tf.rank(i)
        perm = tf.concat([[1,0],tf.range(2,rank)],axis=0)
        it = tf.transpose(i, perm=perm)
        #[Batch, Seq, Blah, Dim] -> [Seq, Batch, Blah, Dim]

        scanned = super().__call__(it, **kwargs)

        rank = tf.rank(scanned)
        perm = tf.concat([[1,0],tf.range(2,rank)],axis=0)
        scanned = tf.transpose(scanned, perm=perm)
        #[Batch, Seq, Blah, Dim] <- [Seq, Batch, Blah, Dim]
        return scanned

# single forward pass version of GRU. Normally we don't use this directly
class GRU_onepass(Can):
    def __init__(self,num_in,num_h):
        super().__init__()
        # assume input has dimension num_in.
        self.num_in,self.num_h = num_in, num_h
        self.wz = Dense(num_in+num_h,num_h,stddev=1,mean=-1) # forget less
        self.wr = Dense(num_in+num_h,num_h,stddev=1)
        self.w = Dense(num_in+num_h,num_h,stddev=1)
        self.incan([self.wz,self.wr,self.w])
        # http://colah.github.io/posts/2015-08-Understanding-LSTMs/

    def __call__(self,i):
        # assume hidden, input is of shape [batch,num_h] and [batch,num_in]
        hidden = i[0]
        inp = i[1]
        wz,wr,w = self.wz,self.wr,self.w
        dims = tf.rank(inp)
        c = tf.concat([hidden,inp],axis=dims-1)
        z = tf.sigmoid(wz(c))
        r = tf.sigmoid(wr(c))
        h_c = tf.tanh(w(tf.concat([hidden*r,inp],axis=dims-1)))
        h_new = (1-z) * hidden + z * h_c
        return h_new

# GRU2 as reported in *Gate-Variants of Gated Recurrent Unit (GRU) Neural Networks*
# the gates are now only driven by hidden state.
# mod: removed reset gate.
# conclusion 20171220: GRU2(without reset gate) is almost as good as GRU.
class GRU2_onepass(Can):
    def __init__(self,num_in,num_h,double=False):
        super().__init__()
        # assume input has dimension num_in.
        self.num_in,self.num_h,self.rect = num_in, num_h, Act('tanh')
        if double==False:
            self.w = Dense(num_in+num_h,num_h,stddev=1.5)
            self.wz = Dense(num_h,num_h,stddev=1)
        else:
            c = Can()
            c.add(Dense(num_in+num_h,int(num_h/2),stddev=1.5))
            c.add(Act('lrelu'))
            c.add(Dense(int(num_h/2),num_h,stddev=1.5))
            c.chain()
            self.w = c
            c = Can()
            c.add(Dense(num_h,int(num_h/2),stddev=1.5))
            c.add(Act('lrelu'))
            c.add(Dense(int(num_h/2),num_h,stddev=1.5))
            c.chain()
            self.wz = c
        self.incan([self.wz,self.w])

    def __call__(self,i):
        # assume hidden, input is of shape [batch,num_h] and [batch,num_in]
        hidden = i[0]
        inp = i[1]
        wz,w = self.wz,self.w
        # dims = tf.rank(inp)
        z = tf.sigmoid(wz(hidden))
        h_c = self.rect(w(tf.concat([hidden,inp],axis=1)))
        h_new = (1-z) * hidden + z * h_c
        return h_new

# vanilla RNN
class RNN_onepass(Can):
    def __init__(self,num_in,num_h,nonlinearity=Act('tanh'),stddev=1):
        super().__init__()
        # assume input has dimension num_in.
        self.num_in,self.num_h,self.rect = num_in, num_h, nonlinearity
        self.w = Dense(num_in+num_h,num_h,stddev=stddev)
        self.incan([self.w,self.rect])

    def __call__(self,i):
        # assume hidden, input is of shape [batch,num_h] and [batch,num_in]
        hidden = i[0]
        inp = i[1]
        c = tf.concat([hidden,inp],axis=1)
        w = self.w
        h_new = self.rect(w(c))
        return h_new

# same but with LayerNorm-ed Dense layers
class GRU_LN_onepass(GRU_onepass):
    def __init__(self,num_in,num_h):
        Can.__init__(self)
        # assume input has dimension num_in.
        self.num_in,self.num_h = num_in, num_h
        self.wz = LayerNormDense(num_in+num_h,num_h,bias=True)
        self.wr = LayerNormDense(num_in+num_h,num_h,bias=True)
        self.w = LayerNormDense(num_in+num_h,num_h,bias=True)
        self.incan([self.wz,self.wr,self.w])

# single forward pass version of GRUConv2D.
class GRUConv2D_onepass(GRU_onepass): # inherit the __call__ method
    def __init__(self,num_in,num_h,*args,**kwargs):
        Can.__init__(self)
        # assume input has dimension num_in.
        self.num_in,self.num_h = num_in, num_h
        self.wz = Conv2D(num_in+num_h,num_h,usebias=False,*args,**kwargs)
        self.wr = Conv2D(num_in+num_h,num_h,usebias=False,*args,**kwargs)
        self.w = Conv2D(num_in+num_h,num_h,usebias=False,*args,**kwargs)
        self.incan([self.wz,self.wr,self.w])

# RNN Can generator from cells, similar to tf.nn.dynamic_rnn
def rnn_gen(name, one_pass_class):
    class RNN(Can):
        def __init__(self,*args,**kwargs):
            super().__init__()
            self.unit = one_pass_class(*args,**kwargs)
            def f(last_state, new_input):
                return self.unit([last_state, new_input])
            self.bscan = BatchScanner(f)
            self.incan([self.unit,self.bscan])
        def __call__(self,i,state_shaper=None,**kwargs):
            # given input, what should be the shape of the state?
            s = tf.shape(i)
            r = tf.rank(i)
            if state_shaper is not None:
                print('(RNN) inferring state_shape using state_shaper().')
                state_shape = state_shaper(i,self.unit.num_h)
            else:
                print('(RNN) inferring state_shape from input.')
                state_shape = tf.concat([[s[0]],s[2:r-1],[self.unit.num_h]],axis=0)
                # [batch, timesteps, blah, dim]->[batch, blah, hidden_dim]

            return self.bscan(i,inferred_state_shape=state_shape, **kwargs)
    RNN.__name__ = name
    return RNN

# you know, Despicable Me
RNN = rnn_gen('RNN', RNN_onepass)
GRU = rnn_gen('GRU', GRU_onepass)
GRU2 = rnn_gen('GRU2', GRU2_onepass)
GRULN = rnn_gen('GRULN', GRU_LN_onepass)
GRUConv2D = rnn_gen('GRUConv2D', GRUConv2D_onepass)

# you know, LeNet
class AvgPool2D(Can):
    def __init__(self,k,std,padding='SAME'):
        super().__init__()
        self.k,self.std,self.padding = k,std,padding

    def __call__(self,i):
        k,std,padding = self.k,self.std,self.padding
        return tf.nn.avg_pool(i, ksize=[1, k, k, 1],
            strides=[1, std, std, 1], padding=padding)

class MaxPool2D(AvgPool2D):
    def __call__(self,i):
        k,std,padding = self.k,self.std,self.padding
        return tf.nn.max_pool(i, ksize=[1, k, k, 1],
            strides=[1, std, std, 1], padding=padding)

# you know, He Kaiming
class ResConv(Can): # v2
    def __init__(self,nip,nop,std=1,bn=True):
        super().__init__()
        # create the necessary cans:
        nbp = int(max(nip,nop)/4) # bottleneck
        self.direct_sum = (nip==nop and std==1)
        # if no downsampling and feature shrinking

        if self.direct_sum:
            self.convs = \
            [Conv2D(nip,nbp,1,usebias=False),
            Conv2D(nbp,nbp,3,usebias=False),
            Conv2D(nbp,nop,1,usebias=False)]
            self.bns = [BatchNorm(nip),BatchNorm(nbp),BatchNorm(nbp)]
        else:
            self.convs = \
            [Conv2D(nip,nbp,1,std=std,usebias=False),
            Conv2D(nbp,nbp,3,usebias=False),
            Conv2D(nbp,nop,1,usebias=False),
            Conv2D(nip,nop,1,std=std,usebias=False)]
            self.bns = [BatchNorm(nip),BatchNorm(nbp),BatchNorm(nbp)]

        self.incan(self.convs+self.bns) # add those cans into collection

    def __call__(self,i):
        def relu(i):
            return tf.nn.relu(i)

        if self.direct_sum:
            ident = i
            i = relu(self.bns[0](i))
            i = self.convs[0](i)
            i = relu(self.bns[1](i))
            i = self.convs[1](i)
            i = relu(self.bns[2](i))
            i = self.convs[2](i)
            out = ident+i
        else:
            i = relu(self.bns[0](i))
            ident = i
            i = self.convs[0](i)
            i = relu(self.bns[1](i))
            i = self.convs[1](i)
            i = relu(self.bns[2](i))
            i = self.convs[2](i)
            ident = self.convs[3](ident)
            out = ident+i
        return out

class BatchNorm(Can):
    def __init__(self,nip,epsilon=1e-3): # number of input planes/features/channels
        super().__init__()
        params_shape = [nip]
        self.beta = self.make_bias(params_shape,name='beta',mean=0.)
        self.gamma = self.make_bias(params_shape,name='gamma',mean=1.)
        self.moving_mean = self.make_variable(
            tf.constant(0.,shape=params_shape),name='moving_mean')
        self.moving_variance = self.make_variable(
            tf.constant(1.,shape=params_shape),name='moving_variance')

        self.epsilon = epsilon

    def __call__(self,x):
        BN_DECAY = 0.99 # moving average constant
        BN_EPSILON = self.epsilon

        # actual mean and var used:
        if get_training_state()==True:

            x_shape = x.get_shape() # [N,H,W,C]
            #params_shape = x_shape[-1:] # [C]

            axes = list(range(len(x_shape) - 1)) # = range(3) = [0,1,2]
            # axes to reduce mean and variance.
            # here mean and variance is estimated per channel(feature map).

            # reduced mean and var(of activations) of each channel.
            mean, variance = tf.nn.moments(x, axes)

            # use immediate when training(speedup convergence), perform update
            moving_mean = tf.assign(self.moving_mean,
                self.moving_mean*(1.-BN_DECAY) + mean*BN_DECAY)
            moving_variance = tf.assign(self.moving_variance,
                self.moving_variance*(1.-BN_DECAY) + variance*BN_DECAY)

            mean, variance = mean + moving_mean * 1e-10, variance + moving_variance * 1e-10
        else:
            # use average when testing(stabilize), don't perform update
            mean, variance = self.moving_mean, self.moving_variance

        x = tf.nn.batch_normalization(x, mean, variance, self.beta, self.gamma, BN_EPSILON)
        return x

# layer normalization on last axis of input.
class LayerNorm(Can):
    def __init__(self,nop=None):
        super().__init__()
        # learnable
        self.alpha = self.make_bias([],mean=1)
        self.beta = self.make_bias([],mean=0)

    def __call__(self,x): # x -> [N, C]
        # axis = len(x.get_shape())-1
        axis = tf.rank(x)-1
        # reduced mean and var(of activations) of each channel.
        mean, var = tf.nn.moments(x, [axis], keep_dims=True) # of shape [N,1] and [N,1]
        # mean, var = [tf.expand_dims(k, -1) for k in [mean,var]]
        var = tf.maximum(var,1e-7)
        stddev = tf.sqrt(var)
        # apply
        normalized = self.alpha * (x-mean) / stddev + self.beta
        # normalized = (x-mean)/stddev
        return normalized

class InstanceNorm(LayerNorm): # for images
    def __call__(self, x):
        mean, var = tf.nn.moments(x, [1,2], keep_dims=True)
        var = var + 1e-8
        stddev = tf.sqrt(var)
        normalized = self.alpha * (x-mean) / stddev + self.beta
        return normalized

import tensorflow as tf
import numpy as np
#from .cans import *

# additional cans

def castf32(i):
    return tf.cast(i,tf.float32)

# RBF glimpse
# evaluate RBF functions representing foveal attention mechanism over the input image, given offset.
class Glimpse2D(Can):
    def __init__(self, num_receptors, pixel_span=20):
        super().__init__()
        if num_receptors<1:
            raise NameError('num_receptors should be greater than 0')
        self.num_receptors = nr = num_receptors
        self.pixel_span = ps = pixel_span

        # generate initial positions for receptive fields
        positions = np.zeros((nr,2),dtype='float32')
        w = int(np.ceil(np.sqrt(nr)))
        index = 0
        for row in range(w):
            for col in range(w):
                if index<nr:
                    positions[index,0] = row/(w-1)
                    positions[index,1] = col/(w-1)
                    index+=1
                else:
                    break

        # positions = np.random.uniform(low=-ps/2,high=ps/2,size=(nr,2)).astype('float32')
        positions = (positions - 0.5) * ps * 0.8
        m = tf.Variable(positions,name='means')
        self.weights.append(m)
        self.means = m

        # stddev of receptive fields
        stddevs = (np.ones((nr,1))*ps*0.5*(1/(w-1))).astype('float32')
        s = tf.Variable(stddevs,name='stddevs')
        self.weights.append(s)
        self.stddevs = s

    def shifted_means_given_offsets(self,offsets):
        means = self.means # [num_of_receptor, 2]

        means = tf.expand_dims(means,axis=0) # [1, num_of_receptor, 2]
        offsets = tf.expand_dims(offsets,axis=1) # [batch, 1, 2]

        shifted_means = means + offsets # [batch, num_of_receptor, 2]

        return shifted_means

    def variances(self):
        variances = tf.nn.softplus(self.stddevs)**2 # [num_of_receptor, 1]
        return variances

    def __call__(self,i): # input: [image, offsets]
        offsets = i[1] # offsets [batch, 2]
        images = i[0] # [batch, h, w, c]

        shifted_means =\
            self.shifted_means_given_offsets(offsets)

        variances = self.variances() # [num_of_receptor, 1]
        variances = tf.expand_dims(variances, axis=0)
        # [1, num_of_receptor, 1]

        ish = tf.shape(images) # [batch, h, w, c]

        uspan = castf32(ish[1])
        vspan = castf32(ish[2])

        # UVMap, aka coordinate system
        u = tf.range(start=(-uspan+1)/2,limit=(uspan+1)/2,dtype=tf.float32)
        v = tf.range(start=(-vspan+1)/2,limit=(vspan+1)/2,dtype=tf.float32)
        # U, V -> [hpixels], [wpixels]

        u = tf.expand_dims(u, axis=0)
        u = tf.expand_dims(u, axis=0)

        v = tf.expand_dims(v, axis=0)
        v = tf.expand_dims(v, axis=0)
        # U, V -> [1, 1, hpixels], [1, 1, wpixels]
        # where hpixels = [-0.5...0.5] * image_height
        # where wpixels = [-0.5...0.5] * image_width

        receptor_h = shifted_means[:,:,0:1]
        # [batch, num_of_receptor, 1(h)]
        receptor_w = shifted_means[:,:,1:2]
        # [batch, num_of_receptor, 1(w)]

        # RBF that sum to one over entire x-y plane:
        # integrate
        #   e^(-((x-0.1)^2+(y-0.3)^2)/v) / (v*pi)
        #   dx dy x=-inf to inf, y=-inf to inf, v>0
        # where ((x-0.1)^2+(y-0.3)^2) is the squared distance on the 2D plane

        # UPDATE 20170405: by using SymPy, we got:
        # infitg(exp((-x**2/var + log(1/pi/var)/2)) * exp(-y**2/var +
        # log(1/pi/var)/2)) = 1

        # squared_dist = (smh - u)**2 + (smw - v)**2
        # [batch, num_of_receptor, hpixels, wpixels]

        # density = tf.exp(- squared_dist / variances) / \
        #         (variances * np.pi)
        # [b, n, h, w] / [1, n, 1, 1]
        # should sum to 1

        # optimized on 20170405 and 20170407
        # reduce calculations to a minimum

        oov = 1/variances
        # half_log_one_over_pi_variances = \
        #     - tf.log(variances)*0.5 + (np.log(1/np.pi)* 0.5)
        # log_one_over_pi_var = np.log(1/np.pi) - tf.log(variances)
        # one_over_pi_variances = (1/np.pi) / variances

        # density = tf.exp(\
        #     -(receptor_h-u)**2 / variances + half_log_one_over_pi_variances) * \
        #     tf.exp(\
        #     -(receptor_w-v)**2 / variances + half_log_one_over_pi_variances)

        density_u = tf.exp(\
            -(receptor_h-u)**2 * oov + np.log(1/np.pi)) * oov
        density_v = tf.exp(\
            -(receptor_w-v)**2 * oov) #* sqrt_pi_variances
        # [b, n, h] and [b, n, w]

        # density_u = tf.expand_dims(density_u, axis=3)
        # density_u = tf.expand_dims(density_u, axis=4)
        # density_v = tf.expand_dims(density_v, axis=3)
        # # [b, n, h, 1, 1] and [b, n, w, 1]

        # density = tf.expand_dims(density, axis=4)
        # [b, n, h, w, 1]

        # images = tf.expand_dims(images, axis=1)
        # # [b, h, w, c] -> [b, 1, h, w, c]
        #
        # tmp = images * density_u
        # # [b, 1, h, w, c] * [b, n, h, 1, 1] -> [b, n, h, w, c]
        # tmp = tf.reduce_sum(tmp, axis=[2]) # -> [b, n, w, c]
        # tmp = tmp * density_v # [b, n, w, c] * [b, n, w, 1] -> [b, n, w, c]
        # tmp = tf.reduce_sum(tmp, axis=[2]) # -> [b, n, c]

        # can we transform above into matmul?
        # [b1wc,h] * [bn11,h] -> [bnwc](sum over h)
        # [bnc,w] * [bn1,w] -> [bnc](sum over w)
        # [bnc] -> [b, n, c]
        # tmp = tf.einsum('bhwc,bnh->bnwc',images,density_u)
        # tmp = tf.einsum('bnwc,bnw->bnc',tmp,density_v)

        # turned out Einstein is a genius:
        tmp = tf.einsum('bhwc,bnh,bnw->bnc',images,density_u,density_v)

        # responses = tf.reduce_sum(density * images, axis=[2,3])
        responses = tmp
        # [batch, num_of_receptor, channel]
        return responses

class GRU_Glimpse2D_onepass(Can):
    def __init__(self, num_h, num_receptors, channels, pixel_span=20):
        super().__init__()

        self.channels = channels # explicit
        self.num_h = num_h
        self.num_receptors = num_receptors
        self.pixel_span = pixel_span # how far can the fovea go

        num_in = channels * num_receptors

        self.glimpse2d = g2d = Glimpse2D(num_receptors, pixel_span)
        self.gru_onepass = gop = GRU_onepass(num_in,num_h)
        self.hidden2offset = h2o = Dense(num_h,2)
        # self.glimpse2gru = g2g = Dense(num_in,num_gru_in)

        self.incan([g2d,gop,h2o])

    def __call__(self,i):
        hidden = i[0] # hidden state of gru [batch, dims]
        images = i[1] # input image [NHWC]

        g2d = self.glimpse2d
        # g2g = self.glimpse2gru
        gop = self.gru_onepass
        h2o = self.hidden2offset

        # hidden is of shape [batch, dims], range [-1,1]
        offsets = self.get_offset(hidden) # [batch, 2]

        responses = g2d([images,offsets]) # [batch, num_receptors, channels]
        rsh = tf.shape(responses)
        responses = tf.reshape(responses,shape=(rsh[0],rsh[1]*rsh[2]))

        # responses2 = g2g(responses)
        # responses2 = Act('lrelu')(responses2)
        hidden_new = gop([hidden,responses])
        return hidden_new

    def get_offset(self, hidden):
        # given hidden state of GRU, calculate next step offset
        # hidden is of shape [batch, dims], range [-1,1]
        h2o = self.hidden2offset
        offsets = tf.tanh(h2o(hidden)) # [batch, 2]
        offsets = offsets * self.pixel_span / 2
        return offsets

GRU_Glimpse2D = rnn_gen('GG2D', GRU_Glimpse2D_onepass)

class PermutedConv2D(Can):
    def __init__(self, nip, nop, bottleneck_width, seed, k=3, std=1, *a, **w):
        super().__init__()
        self.nip, self.nop, self.bnw = nip, nop, bottleneck_width
        assert nip >= self.bnw
        assert nop >= nip
        self.num_output_padding = nop - nip

        self.conv = self.add(Conv2D(self.bnw, self.bnw, k=k, std=std, stddev=.5, *a, **w))
        self.seed = seed

        from numpy.random import RandomState
        prng = RandomState(self.seed)

        if self.num_output_padding>0:
            self.io_permtable = prng.randint(self.nip, size = [self.num_output_padding])

        self.offset = prng.randint(self.nip-self.bnw+1)
        self.bo_permtable = prng.randint(self.bnw, size = [self.nop])

    def __call__(self, i):
        # assume [NHWC]
        s = tf.shape(i)

        residual = i

        if self.conv.std>1:
            residual  = AvgPool2D(k=self.conv.k, std=self.conv.std)(residual)

        if self.num_output_padding > 0:
            residual_pad = tf.gather(residual, self.io_permtable, axis=3)
            residual = tf.concat([residual, residual_pad], axis=3)

        piece = i[:,:,:, self.offset: self.offset + self.bnw]
        piece = self.conv(piece)
        piece = Act('lrelu')(piece)

        scattered = tf.gather(piece, self.bo_permtable, axis=3)
        result = scattered + residual*0.707
        return result

import tensorflow as tf
import os
_SESSION = None
_TRAINING = True

def flatten(items,enter=lambda x:isinstance(x, list)):
    # http://stackoverflow.com/a/40857703
    """Yield items from any nested iterable; see REF."""
    for x in items:
        if enter(x):
            yield from flatten(x)
        else:
            yield x

# borrowed from Keras
def get_session():
    """Returns the TF session to be used by the backend.
    If a default TensorFlow session is available, we will return it.
    Else, we will return the global Keras session.
    If no global Keras session exists at this point:
    we will create a new global session.
    Note that you can manually set the global session
    via `K.set_session(sess)`.
    # Returns
        A TensorFlow session.
    """
    global _SESSION
    if tf.get_default_session() is not None:
        session = tf.get_default_session()
    else:
        if _SESSION is None:
            if not os.environ.get('OMP_NUM_THREADS'):
                config = tf.ConfigProto(allow_soft_placement=True)
            else:
                nb_thread = int(os.environ.get('OMP_NUM_THREADS'))
                config = tf.ConfigProto(intra_op_parallelism_threads=nb_thread,
                                        allow_soft_placement=True)
            _SESSION = tf.Session(config=config)
        session = _SESSION
    # if not _MANUAL_VAR_INIT:
    #     _initialize_variables()
    return session


def set_session(session):
    """Sets the global TF session.
    """
    global _SESSION
    _SESSION = session

def set_training_state(state=True):
    global _TRAINING
    _TRAINING = state

def get_training_state():
    global _TRAINING
    return _TRAINING

def set_variable(value,variable=None):
    """Load some value into session memory by creating a new variable.
    If an existing variable is given, load the value into the given variable.
    """
    sess = get_session()
    if variable is not None:
        assign_op = tf.assign(variable,value)
        sess.run([assign_op])
        return variable
    else:
        variable = tf.Variable(initial_value=value)
        sess.run([tf.variables_initializer([variable])])
    return variable

def get_variables_of_scope(collection_name,scope_name):
    var_list = tf.get_collection(collection_name, scope=scope_name)
    return var_list

def ph(shape,*args,**kwargs):
    return tf.placeholder(tf.float32,shape=[None]+shape,*args,**kwargs)

def gvi():
    return tf.global_variables_initializer()

import tensorflow as tf

eps = 1e-8

def loge(i):
    return tf.log(i+eps)

def one_hot_accuracy(pred,gt):
    correct_vector = tf.equal(tf.argmax(pred,1), tf.argmax(gt,1))
    acc = tf.reduce_mean(tf.cast(correct_vector,tf.float32))
    return acc

def class_accuracy(pred,lbl):
    correct_vector = tf.equal(tf.argmax(pred,1,output_type=tf.int32), lbl)
    acc = tf.reduce_mean(tf.cast(correct_vector,tf.float32))
    return acc

def mean_softmax_cross_entropy(pred,gt):
    return tf.reduce_mean(softmax_cross_entropy(pred,gt))

def softmax_cross_entropy(pred,gt):
    # tf r1.0 : must use named arguments
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=gt)

def cross_entropy_loss(pred,gt): # last dim is one_hot
    return - tf.reduce_mean(tf.reduce_sum(loge(pred) * gt, axis=tf.rank(pred)-1))

def binary_cross_entropy_loss(pred,gt,l=1.0): # last dim is 1
    return - tf.reduce_mean(loge(pred) * gt + l * loge(1.-pred) * (1.-gt))

def sigmoid_cross_entropy_loss(pred,gt): # same as above but more stable
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=gt)

def mean_sigmoid_cross_entropy_loss(pred,gt): # same as above but more stable
    return tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=gt))


