# Ring-Loss-Keras
Keras implementation of Ring Loss : Convex Feature Normalization for Face Recognition. Based on https://arxiv.org/abs/1803.00130

This paper highlights the importance of feature normalization in feature space for enhanced learning, unlike earlier methods e.g - L2 Constrained Softmax

## Getting Started
Install Keras and Python.
Download ringloss-keras.py to your working directory. 

## Usage
Initialize a Ring Loss layer and call the layer with your input feature

```
lambda_ring = 1.0
ring_loss = Ring_Loss(radius = 1.0, name = 'ring_loss')(feature)
```

Finally, compile your model with softmax + ringloss . e.g.
```
num_classes = 10
x_in = Dense(num_classes, name = 'final_layer', kernel_initializer = 'he_normal')(feature) 
output = Activation('softmax', name = 'softmax_out')(x_in)
    
#compile model with ring loss + softmax    
model.compile(loss = {'softmax_out' : 'categorical_crossentropy', 'ring_loss': identity_loss}, optimizer= opt,  metrics = ['accuracy'], loss_weights=[1,lambda_ring]) 
    
```

## Training

Pass a random output for ring loss during the batch data generation to satisfy the outputs.

```
random_y_train = np.random.rand(batch_size,1)
x_label, y_label =  [data], [y_trues, random_y_train]
```
