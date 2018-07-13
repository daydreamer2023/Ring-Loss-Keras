# Ring-Loss-Keras
Keras implementation of Ring Loss : Convex Feature Normalization for Face Recognition. Based on https://arxiv.org/abs/1803.00130

## What?
This paper highlights the importance of feature normalization in feature space for better clustering, unlike earlier methods (e.g - L2 Constrained Softmax). The authors have designed a novel loss called Ring Loss to optimize over this norm constraint.

## Why?
The direct approach to feature normalization through the hard normalization operation results in a non-convex formulation. Instead, Ring loss applies soft normalization, where it gradually learns to constrain the norm to the scaled unit circle while preserving convexity leading to more robust features.

## Getting Started
Install Keras and Python.
Download ringloss-keras.py to your working directory. 

## Usage
Initialize a Ring Loss layer and call the layer with your input feature

```
lambda_ring = 0.1 #Loss Weight Range : 0.1-0.5 to ensure that the ring loss doesn't dominate the optimization process

ring_loss = Ring_Loss(radius = 1.0, loss_type = 'smoothl1', name = 'ring_loss')(feature)
#--> loss_types - 'cauchy', 'geman', 'smoothl1', 'l2' - 'smoothl1' is default
#--> shape of feature - (batch_size, feature_dims)

```

Finally, compile your model with softmax + ringloss

```
#init number of classes
num_classes = 10

#init final fully connected layer after the feature layer
x_final = Dense(num_classes, name = 'final_layer', kernel_initializer = 'he_normal')(feature) 
output = Activation('softmax', name = 'softmax_out')(x_in)
    
#compile model with joint loss - (softmax loss + lambda_ring * ring loss)
model.compile(loss = {'softmax_out' : 'categorical_crossentropy', 'ring_loss': identity_loss}, optimizer = opt,  metrics = ['accuracy'], loss_weights=[1,lambda_ring])     
```

## Training

Pass a random output for ring loss during the batch data generation to satisfy the outputs.

```
random_y_train = np.random.rand(batch_size,1)
x_label, y_label =  [data], [y_trues, random_y_train]
```

## Contact
To ask questions or report issues, please open an issue on the [issues tracker](https://github.com/vsatyakumar/Ring-Loss-Keras/issues).
