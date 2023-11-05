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

# Comparation between models:

## You can run the following scripts to generate results:

```python
python3 VGG_exe_all.py  
```
```python
python3 Alexnet_exe_all.py  
```
```python
python3 customized_cnn_exe_all.py  
```

## You can go to `code/comparison` to see the winner of each round.

## Round 1:
## model: AlexNet_256 + Activation Function on FC (Sigmoid, Tanh, Relu) + Optimizer (Adam, SGD)
### select the winner

## model: AlexNet_4096 + Activation Function on FC (Tanh, Relu) + Optimizer (Adam, SGD)
### select the winner

### then we compare AlexNet-4096 vs Alec-256, and select the winner. -> winner of Round 1

## Round 2:
## model: VGG16_4096 + Optimizer (Adam, Adam_amsgrad)
### select the winner -> winner of Round 2

## Round 3:
## model: Customized_cnn + Optimizer (Adam_amsgrad, SGD) + LR Scheduler
### select the winner -> winner of Round 3

## Round 4:
## Round 1 vs Round 2 vs Round 3, and we should also consider "efficiency"
### select the final winner

## Others:
### compare early stop param v.s. non-early stop params, empirical loss v.s. general loss

## Advanced:
### Continue Learning