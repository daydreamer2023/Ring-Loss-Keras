# Ring-Loss-Keras
Keras implementation of Ring Loss : Convex Feature Normalization for Face Recognition. Based on https://arxiv.org/abs/1803.00130

## Getting Started
Install Keras and Python.
Download ringloss.py to your working directory. 

## Usage
Initialize a RingLoss layer
```
ring_loss_layer = Ring_Loss(init_radius = 1.0, name = 'ring_loss')(feature_dlib)
```
During model creation, call the layer with your input feature
```
ringloss = ring_loss_layer(feature) # your feature should be (batch_size x feat_size)
```
Finally, compile model combined with softmax . e.g.
```
model.compile(loss = {'softmax_output' : 'categorical_crossentropy', 'ring_loss': identity_loss}, optimizer= opt,  metrics = ['accuracy'], loss_weights=[1,lambda_ring]) 
    
```

## Training
