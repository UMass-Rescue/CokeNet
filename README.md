# SodaNet
The following repository hosts the soda-net modules which is a binary classifier for distinguishing whether a network contains soda bottles (in particular pepsi or coca-cola based products) over others. 

## Network Architecture

Model: "sequential"
_________________________________________________________________
#### Layer (type)                 #### Output Shape             ####  Param 
=================================================================
conv2d (Conv2D)              (None, 78, 78, 48)        3648      
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 39, 39, 48)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 20, 20, 96)        115296    
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 10, 10, 96)        0         
_________________________________________________________________
batch_normalization (BatchNo (None, 10, 10, 96)        384       
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 5, 5, 192)         460992    
_________________________________________________________________
batch_normalization_1 (Batch (None, 5, 5, 192)         768       
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 3, 3, 192)         921792    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 1, 1, 192)         0         
_________________________________________________________________
batch_normalization_2 (Batch (None, 1, 1, 192)         768       
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 1, 1, 256)         1229056   
_________________________________________________________________
batch_normalization_3 (Batch (None, 1, 1, 256)         1024      
_________________________________________________________________
flatten (Flatten)            (None, 256)               0         
_________________________________________________________________
dense (Dense)                (None, 512)               131584    
_________________________________________________________________
dense_1 (Dense)              (None, 256)               131328    
_________________________________________________________________
dense_2 (Dense)              (None, 128)               32896     
_________________________________________________________________
    dense_3 (Dense)              (None, 2)              258     
_________________________________________________________________
Total params: 3,029,794
Trainable params: 3,028,322
Non-trainable params: 1,472
_________________________________________________________________

## Usage

Method Name | #Inputs | Description
--- | --- | ----
get_model | None | Returns a copy of the model. 
set_model | Pytorch Model | Reinsantiates a new temporary model (optional, and usage limited to advanced exploration)
predict | None | Returns the predicted score (between 0 and 1) for a particular image or set of images. 
evaluate | csv file path (optional) | Returns a binary prediction alongside saving the results in the csv file if the csv path was set. 
retrain | folder path for positive, negative images for training | Retrains the network on user-defined images for a more hybrid and customized model