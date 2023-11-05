# Group 6 : Emotion Recognition from Facial Expressions
## Author: Haohua Feng, PinKuan Hsieh
# Abstract:
The project's objective is to train a CNN (Convolutional Neural Network) model capable of accurately distinguishing human facial expressions. Our target is to achieve a minimum accuracy rate of 95%. The ultimate aim is to implement this model in user interface design and mental health monitoring applications.

# Project Requirement
1. Collect and preprocess the facial expression dataset. (5)
2. Implement a basic emotion recognition model using CNNs. (20)
3. Optimize the model for better performance. (45)
4. Evaluate the model’s performance and integrate it with a real-time webcam feed. (30)

# 1. Dataset:
We utilize the FER-2013 dataset, comprising a training subset and a testing subset. Our approach involves training a CNN on the training set and evaluating its performance using the testing set.

# 2. Model Description:

# 3. Loss Function:

# 4. Optimization Algorithm:

# 5. Metrics and Experimental Results:

# 6. Contributions and GitHub:

# comparation between models:
## model: AlexNet + Activation Function on FC + Optimizer
## 1. AlexNet-256 + sigmoid + Adam (Win) v.s. AlexNet-256 + sigmoid + SGD
## 2. AlexNet-256 + Tanh + Adam (Win) v.s. AlexNet-256 + Tanh + SGD
## 3. AlexNet-256 + Relu + Adam v.s. AlexNet-256 + Relu + SGD (?)
## => AlexNet-256 + sigmoid + Adam v.s. AlexNet-256 + Tanh + Adam v.s. (3.) (?)
## so we should use Adam + (Tanh or Relu). Then we move to 4096 layer
## 3. AlexNet-4096 + Relu + Adam (Win) v.s. AlexNet-4096 + Tanh + Adam
## then we compare AlexNet-4096 vs Alec-256, and select the winner.

# model: VGG16 + Adam v.s. VGG16 + Adam with amsgrad

# model: customized_cnn + Adam with amsgraddam + LR scheduler v.s. customized_cnn + SGD + LR scheduler

# compare early stop param v.s. non-early stop params

# run the following scripts to compare results:

```python
python3 VGG_exe_all.py  
```
```python
python3 Alexnet_exe_all.py  
```
```python
python3 customized_cnn_exe_all.py  
```