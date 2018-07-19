import keras
from keras.engine.topology import Layer
from keras import backend as K
from keras.initializers import Constant
import tensorflow as tf


def identity_loss(y_true, y_pred):
    return y_pred
    
def huber_ring_loss(x, ring_norm, HUBER_DELTA = 1.0):
    l2_norm = K.sqrt(K.sum(K.square(x), axis = -1))
    error = l2_norm - ring_norm
    huber_loss = K.switch(error < HUBER_DELTA, 0.5 * error ** 2, HUBER_DELTA * (error - 0.5 * HUBER_DELTA))
    return  huber_loss

def squared_ring_loss(x, ring_norm):
    #calculate l2 norm of features
    l2_norm = K.sqrt(K.sum(K.square(x), axis = -1))
    return 0.5 * K.square(l2_norm - ring_norm) 

def cauchy_ring_loss(x, ring_norm, scale_factor = 2.3849):
    alpha = 0.5 * (scale_factor ** 2)
    #calculate l2 norm of features
    l2_norm = K.sqrt(K.sum(K.square(x), axis = -1))
    return alpha * K.log(1.0 + K.square((l2_norm - ring_norm) / scale_factor))

def geman_ring_loss(x, ring_norm, alpha = 0.5):
    #calculate squared error
    l2_norm = K.sqrt(K.sum(K.square(x), axis = -1))
    squared_error = K.square(l2_norm - ring_norm)
    return tf.divide(alpha * squared_error, squared_error + (2.0 * alpha))

    
#Ring Loss - https://arxiv.org/abs/1803.00130
class Ring_Loss(Layer):
    def __init__(self, radius = 1.0, loss_type = 'l2', **kwargs):
        self.radius = radius
        self.huber_delta = 1.0
        self.cauchy_scale_factor = 2.3849 #cauchy constant from - http://webdiis.unizar.es/~jcivera/papers/concha_civera_ecmr15.pdf
        self.geman_alpha = 0.5 #geman constant from the same source as above.
        self.loss_type = loss_type
        super(Ring_Loss, self).__init__(**kwargs)

    def build(self, input_shape):
        self.var_shape = (1,)
        #init norm value
        self.ring_norm = self.add_weight(name='ring_norm',
                                     shape=self.var_shape,
                                     initializer = Constant(self.radius), 
                                     dtype = K.floatx(),
                                     trainable = True)
        super(Ring_Loss, self).build(input_shape)

    def call(self, x):
        if self.loss_type == 'squared':
            #calculate L2 ring loss
            self.ring_loss = squared_ring_loss(x, self.ring_norm)

        elif self.loss_type == 'cauchy':
            #calculate cauchy ring loss
            self.ring_loss = cauchy_ring_loss(x, self.ring_norm, scale_factor = self.cauchy_scale_factor)
            
        elif self.loss_type == 'geman':
            #calculate geman-mcclure ring loss
            self.ring_loss = geman_ring_loss(x, self.ring_norm, alpha = self.geman_alpha)    
                     
        elif self.loss_type == 'huber': 
            #calculate Smooth-L1/Huber ring loss
            self.ring_loss = huber_ring_loss(x, self.ring_norm, HUBER_DELTA = self.huber_delta)
        else:
            print("Invalid Loss Name specified. Available options are : 'squared', 'huber', 'cauchy' and 'geman'. Continuing with default loss - 'huber'")
            self.ring_loss = huber_ring_loss(x, self.ring_norm, HUBER_DELTA = self.huber_delta)
        
        return self.ring_loss

    def get_config(self):
        config = {'radius': self.radius, 'loss_type' : self.loss_type}
        base_config = super(Ring_Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0],1)
        return output_shape
