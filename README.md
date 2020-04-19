# SodaNet
The following repository hosts the soda-net modules which is a binary classifier for distinguishing whether a network contains soda bottles (in particular pepsi or coca-cola based products) over others. 

## Network Architecture

**Model: "sequential"**
_________________________________________________________________
Layer (type) | Output Shape | Parameters
--- | --- | ---
conv2d (Conv2D) | (None, 78, 78, 48) | 3648
max_pooling2d (MaxPooling2D) | (None, 39, 39, 48) | 0
conv2d_1 (Conv2D) | (None, 20, 20, 96) | 115296 
max_pooling2d_1 (MaxPooling2) | (None, 10, 10, 96) | 0
batch_normalization |  (None, 10, 10, 96) | 384
conv2d_2 (Conv2D) | (None, 5, 5, 192) | 460992
batch_normalization_1 | (None, 5, 5, 192) | 768
conv2d_3 (Conv2D)  | (None, 3, 3, 192) | 921792
max_pooling2d_2  | (None, 1, 1, 192) | 0
batch_normalization_2 | (None, 1, 1, 192) | 768
conv2d_4 (Conv2D) | (None, 1, 1, 256) | 1229056
batch_normalization_3 | (None, 1, 1, 256) | 1024
flatten (Flatten) | (None, 256) | 0
dense (Dense) | (None, 512) | 131584
dense_1 (Dense) | (None, 256) | 131328
dense_2 (Dense) | (None, 128) | 32896
dense_3 (Dense) | (None, 2) | 258     
_________________________________________________________________
Total params: 3,029,794
Trainable params: 3,028,322
Non-trainable params: 1,472
___________________________________________________________

## Usage

Method Name | #Inputs | Description
--- | --- | ----
get_model | None | Returns a copy of the model. 
set_model | Pytorch Model | Reinsantiates a new temporary model (optional, and usage limited to advanced exploration)
predict | None | Returns the predicted score (between 0 and 1) for a particular image or set of images. 
evaluate | csv file path (optional) | Returns a binary prediction alongside saving the results in the csv file if the csv path was set. 
retrain | folder path for positive, negative images for training | Retrains the network on user-defined images for a more hybrid and customized model
load_image | image variable (Preferable: numpy/cv2/Image) | Serves image to the network for making predictions
load_image_from_file | folder path containing image | Loads image from the folder to be served to the network for making predictions

## Usage Examples



```python
# Required packages (general)
from sodanet_model import SodaModel
from matplotlib.image import imread
from utilities import compute_accuracy_labelwise

# Demo-specific packages (not required for the working of this module) 
from matplotlib import pyplot as plt
import pandas as pd
```
```python
# Prediction from single image variable
image_path = 'dataset/custom/temp_coke/5a86e60cd0307215038b4797.jpg'
im = imread(image_path)
plt.imshow(im)
plt.show()
```


![png](output_2_0.png)



```python
# Loading the sodanet module
model = SodaModel()
```


```python
# Making predictions for a single image variable
model.load_image(im)
predicted, im_ret = model.evaluate()
print ("Predicted: 0-Not Coke, 1-Coke : ", predicted) # 0: Not Coke, 1: Coke
plt.imshow(im_ret)
plt.show()
```

    Predicted: 0-Not Coke, 1-Coke :  [1]
    


![png](output_4_1.png)



```python
# Resizing single image file
import cv2
import os
import numpy as np

def resize_image_single(im, output_shape=160):
    resized = cv2.resize(im, (output_shape, output_shape), interpolation=cv2.INTER_AREA)
    resized[resized<0] = 0
    if resized.shape == (output_shape, output_shape, 4):
        resized = resized[:, :, :3]
    return cv2.normalize(resized, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, 
                                 dtype = cv2.CV_32F).astype(np.uint8)
```


```python
# Prediction from multiple image variable derived from a directory
max_files = 10
base_path = 'dataset/custom/temp_coke/'
files = os.listdir(base_path)[:max_files]
im = np.array([resize_image_single(imread(os.path.join(base_path, file))) for file in files])
model.load_image(im)
predicted, ret_im = model.evaluate()
print (predicted)
print ("Is returned img array same as the original ? : {}".format(np.testing.assert_almost_equal(im, ret_im) is None))
```

    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Is returned img array same as the original ? : True
    

```python
# Prediction for a batch of files plus as well as displaying what goes inside the csv file
model.load_image_from_file('dataset/custom/temp_not_coke')
_, results_tray = model.evaluate(output_csv_path='dataset/output/not_coke_pred.csv', mode='w')
print ("Rightly-classified Accuracy  = {}".format(compute_accuracy_labelwise(results_tray, 1)))
display(pd.read_csv('dataset/output/not_coke_pred.csv'))
```

**Rightly-classified Accuracy**  = 0.7236842105263158
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2997084866_a6f8749434_o.jpg</th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3023311841_c26dac5dfd_o.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3023311851_6f933bfb3a_o.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3023311879_a4e284202b_o.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3024147898_082442f05a_o.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3025918647_703a4de870_o.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>3593355893_a52329d7f7_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>3593912700_e5704573b3_b.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>3609363287_2a698128fe_o.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>73</th>
      <td>3617371187_4db81e2829_o.jpg</td>
      <td>0</td>
    </tr>
    <tr>
      <th>74</th>
      <td>cutest-puppy-dog-pictures-coverimage.jpg</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 2 columns</p>
</div>



```python
model.load_image_from_file('dataset/custom/temp_coke')
_, results_tray = model.evaluate(output_csv_path='dataset/output/coke_pred.csv', mode='w')
print ("Rightly-classified Accuracy = {}".format(compute_accuracy_labelwise(results_tray, 0)))
display(pd.read_csv('dataset/output/coke_pred.csv'))
```

**Rightly-classified Accuracy** = 1.0
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.jpg</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1030426_d94dfc35f3_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>109941551_962c4dcdea_b.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13294174925_d34609da50_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>138554565_77e5ae971c_b.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>142828186_ab849fa6b8_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>960x0.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>59</th>
      <td>9864241_ac1b13b949_b.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Coca-cola_1040.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>coke-corona-virus-today-main-200226-v2_1e37aa5...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Coke_PageProperties.jpg</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>63 rows × 2 columns</p>
</div>



```python
# Attempting to increase accuracy through training
dir_coke = 'dataset/custom/temp_coke'
dir_not_coke = 'dataset/custom/temp_not_coke'
validation_coke, validation_not_coke = '', ''
default_transformations_coke, default_transformations_not_coke = ['perform_random_affine_transform', 'flip_rotate', 
                                                                 'add_gaussian_noise'], []
model_params = {'max_epochs': 15, 'external_model_path': 'sodanet/additional_model'}

model.retrain(dir_coke, dir_not_coke, validation_coke, validation_not_coke, default_transformations_coke, 
             default_transformations_not_coke, model_params)
```

    Train on 1320 samples
    Epoch 1/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1753 - accuracy: 0.9939
    Epoch 2/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1713 - accuracy: 0.9947
    Epoch 3/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1669 - accuracy: 0.9970
    Epoch 4/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1648 - accuracy: 0.9977
    Epoch 5/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1639 - accuracy: 0.9985
    Epoch 6/15
    1320/1320 [==============================] - 20s 15ms/sample - loss: 0.1627 - accuracy: 0.9992
    Epoch 7/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1615 - accuracy: 0.9992
    Epoch 8/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1607 - accuracy: 0.9992
    Epoch 9/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1600 - accuracy: 0.9992
    Epoch 10/15
    1320/1320 [==============================] - 20s 16ms/sample - loss: 0.1598 - accuracy: 1.0000
    Epoch 11/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1600 - accuracy: 1.0000
    Epoch 12/15
    1320/1320 [==============================] - 22s 16ms/sample - loss: 0.1601 - accuracy: 1.0000
    Epoch 13/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1601 - accuracy: 1.0000
    Epoch 14/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1601 - accuracy: 1.0000
    Epoch 15/15
    1320/1320 [==============================] - 21s 16ms/sample - loss: 0.1601 - accuracy: 1.0000
    


```python
# Prediction for a batch of files
model.load_image_from_file('dataset/custom/temp_not_coke')
_, results_tray = model.evaluate(output_csv_path='dataset/output/not_coke_pred.csv', mode='w')
print ("Rightly-classified Accuracy  = {}".format(compute_accuracy_labelwise(results_tray, 1)))
print (results_tray)
display(pd.read_csv('dataset/output/not_coke_pred.csv'))
```

**Rightly-classified Accuracy**  = 0.8026315789473685 (Improvement of over 8%)!!
   
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2997084866_a6f8749434_o.jpg</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3023311841_c26dac5dfd_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3023311851_6f933bfb3a_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3023311879_a4e284202b_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3024147898_082442f05a_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3025918647_703a4de870_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>3593355893_a52329d7f7_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>3593912700_e5704573b3_b.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>72</th>
      <td>3609363287_2a698128fe_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>73</th>
      <td>3617371187_4db81e2829_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>74</th>
      <td>cutest-puppy-dog-pictures-coverimage.jpg</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>75 rows × 2 columns</p>
</div>



```python
model.load_image_from_file('dataset/custom/temp_coke')
_, results_tray = model.evaluate(output_csv_path='dataset/output/coke_pred.csv', mode='w')
print ("Rightly-classified Accuracy = {}".format(compute_accuracy_labelwise(results_tray, 0)))
print (results_tray)
display(pd.read_csv('dataset/output/coke_pred.csv'))
```

**Rightly-classified Accuracy** = 1.0 (Recall remains 100% while we still get improvement in F1-Score)
  
<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.jpg</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1030426_d94dfc35f3_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>109941551_962c4dcdea_b.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13294174925_d34609da50_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>138554565_77e5ae971c_b.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>142828186_ab849fa6b8_o.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>58</th>
      <td>960x0.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>59</th>
      <td>9864241_ac1b13b949_b.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>60</th>
      <td>Coca-cola_1040.jpg</td>
      <td>1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>coke-corona-virus-today-main-200226-v2_1e37aa5...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>62</th>
      <td>Coke_PageProperties.jpg</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>63 rows × 2 columns</p>
</div>