# Ring-Loss-Keras
Keras implementation of Ring Loss : Convex Feature Normalization for Face Recognition. Based on https://arxiv.org/abs/1803.00130

## Getting Started
Install Keras and Python.
Download ringloss.py to your working directory. 

## Usage
Initialize a Ring Loss layer and call the layer with your input feature

```
ring_loss = Ring_Loss(init_radius = 1.0, name = 'ring_loss')(feature)
```

Finally, compile your model with softmax + ringloss . e.g.
```
model.compile(loss = {'softmax_output' : 'categorical_crossentropy', 'ring_loss': identity_loss}, optimizer= opt,  metrics = ['accuracy'], loss_weights=[1,lambda_ring]) 
    
```

## TRAINING

Pass a random output for ring loss during the batch data generation to satisfy the outputs.

'''
random_y_train = np.random.rand(batch_size,1)
x_label, y_label =  [data], [y_trues, random_y_train]
'''
