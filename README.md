# Trustworthy AI

## Overview
This is an implemetation project of "Evil Model" mentioned in Zhi Wang(2021). 
<https://arxiv.org/pdf/2107.08590.pdf?>

This program aims to help Ernst & Young clients learn about the latest malware attack methods and understand the importance of information security maintenance.

## Methodology
In the paper EvilModel: Hiding Malware Inside of Neural Network Models, a technology called fast substitution is mentioned, which converts the virus code into IEEE-754 standard 32-bit floating point numbers and replaces the parameters. The method is embedded in the deep learning model to avoid detection by anti-virus software. 
![image](https://github.com/show0117/Trustworthy_AI/blob/main/Embedding_Technique.png)

When the user uses a specific program, the virus can be extracted from the model and executed. The actual process is as follows.
![image](https://github.com/show0117/Trustworthy_AI/blob/main/Malware_Infection_Process.png)

In this project, our ultimate goal is to embed the simulated virus Virus.vbs file we prepared into our Infected_Model.pt file, and record the infected neurons in the Infected_Neurons.csv file for subsequent virus detection extraction.

## Model Training
In **trustedAI_AlexNet.ipynb**, We select AlexNet for a more slight implementation by *PyTorch*. We use the fashion_mnist dataset released by *Tensorflow* for training. This is a dataset that contains various fashion apparel, such as hats, shoes, clothes, etc. Since the dataset is quite large, we read it by linking to the cloud data. Conduct 10 times of training on the fully connected layer, while observing the accuracy and loss function of the verification group. The remaining network layers remain unchanged. The results are as follows. In the end, an accuracy rate of nearly 90% can be achieved.
![image](https://github.com/show0117/Trustworthy_AI/blob/main/Training_Result.png)

## Model Embedding
In **trustedAI_mainCode.ipynb**, we convert the program code in the virus into 32-bit data format according to the method in the paper and add \x3c before each group of data in groups of three, and then convert it into the corresponding IEEE-754 standard 32-bit A floating point number. In this demo file, you can see the whole process of the malware embedding by the following steps.

### Finding the fully connected layer parameters
Since the convolution layer can extract image features, we first find the three fully connected layers of the model and replace them first without significantly affecting the model performance.

### Searching for replaceable neurons
We have previously converted the virus's code into floating point numbers. At this time, we search the parameters of the connection layer to find the neurons that are within 0.0001 of the virus's floating point numbers and replace them. In this way, the prediction of the model The force will not be affected too much.

### Recording the coordinates of the virus
We record the locations of infected neurons and save them as **Infected_Neurons.csv** file to facilitate subsequent removal of the virus from the model.

### Saving the infected model
Save the model weights embedded with the virus into the **Infected_Model.pt** file.

## Model Decoding
Beware of the **run_classifier.py** file. This file is in charge of decoing the infected AlexNet model into a malware and forcefully execute it!!!!! In other word, if user is unware of the infected deep models they download by other open platforms, when they excute the programs, their computers will be infected. 
![image](https://github.com/show0117/Trustworthy_AI/blob/main/Infection.png)

## trustworthyAI_demo.mp4
This is a guidance video for the real implementation process of our program. 

