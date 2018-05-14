import keras
from keras.engine.topology import Layer
from keras import backend as K

class Ring_Loss(Layer):
    def __init__(self, radius = 1.0, **kwargs):
        self.radius = radius
        super(Ring_Loss, self).__init__(**kwargs)

    def build(self, input_shape):
        self.var_shape = (1,)
        #init radius weight value
        self.ring_norm = self.add_weight(name='ring_norm',
                                     shape=self.var_shape,
                                     initializer = Constant(self.radius), 
                                     dtype = K.floatx(),
                                     trainable = True)
        
        super(Ring_Loss, self).build(input_shape)

    def call(self, x):
        #calculate l2 norm of features
        self.l2_norm = K.sqrt(K.sum(K.square(x), axis = -1))
        
        #calculate ring loss
        self.ring_loss = K.square(self.l2_norm - self.ring_norm) / 2.0
        
        return self.ring_loss

    def get_config(self):
        config = {'radius': self.radius}
        base_config = super(Ring_Loss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0],1)
        return output_shape
    
    

