# Audio Deepfake Detection

I trained a Convolutional Neural Net for the detection of audio deepfakes.

### Repo Guide

- assets/ - contains images to display in this Readme
- cnn/ - contains the convolutional neural net architecture, built in PyTorch
- data/ - contains meta data, as well as functions for importing and handling data
- models/ - contains trained models
- tests/ - contains unit tests
- analyze.ipynb - contains the audio clip analyze tool (see below for more)
- eval.py - contains functions related to model evaluation
- hyperparameter_tuning.txt - contains the output from a hyperparameter tuning run
- train.ipynb - contains the code for training my CNN's, as well as some evaluation of the final model I trained 


## Data Collection & Splitting

Using the 'In-the-Wild' dataset of audio deepfake data from kaggle -- [link](https://www.kaggle.com/datasets/abdallamohamed312/in-the-wild-dataset/data)

See 'data/data_meta.csv' for data on the samples and their labels.

Stats:
- 31,779 total audio samples of varying lengths, with a sample rate of 16kHz
  - 19,963 real
  - 11,816 fake
  - An average length of 4.28 seconds
- 54 total speakers
  - these are real people, and the dataset includes both real and spoofed audio of them speaking

Splitting data:


Evaluation:
- One speaker (Alec Guiness) was removed - he will be our final evaluation set
- This way, the model will have never heard any of his audio before
  - Alec Guiness:
    - 1907 real clips
    - 1718 fake clips

  - This leaves (for our train and test sets):
    - 28,154 clips
      - 18,056 real
      - 10,098 fake

Train & Test:
- I want to split them up so that any individual speaker is only in one set
- I'll try to split it 80/20
- Train:
  - 20,738 clips (73%)
    - 12,745 real (61%)
    - 7,993 fake (39%)
- Test:
  - 7,416 clips (27%)
    - 5,311 real (71%)
    - 2,105 fake (29%)



## Building the Convolutional Neural Net

I need to calculate the dimensions of the CNN.

The network will be composed of a 2D Convolution, Batch Normalization, a ReLU, 2D Max Pooling, then a Dropout layer

Then, I'll have a Linear layer after that

Inputs are of size [batch_size, 1, 128, 80]

##### 2D Convolution:

stride = 1, padding = 0, dilation = 1, kernel = (3 x 3)

![](/assets/images/conv2d_output_shape.png?raw=True "Output Size (from PyTorch docs)")

Hout = 128 - 2 - 1 + 1 == 126

Wout = 80 - 2 - 1 + 1 == 78

Output: [batch_size, num_channels, 126, 78]

##### Max Pooling

Input: [batch_size, 16, 126, 78]

stride=2, padding=0, dilation=1, kernel = (2 x 2)

![](/assets/images/maxpool2d_output_shape.png?raw=True "Output Size (from PyTorch docs)")

Hout = (126 - 1 - 1) / 2 + 1 = 63

Wout = (78 - 1 - 1) / 2 + 1 = 39

Output: [batch_size, num_channels, 63, 39]

##### Linear Layer

Now, we'll need to flatten the inputs for the linear layer.

Their dimensions will be num_channels * 63 * 39

Output will be of dimension 2

## Training

I ran a hyperparameter grid search, with early stopping enabled, to try to identify the right mix of hyperparameters.

##### Hyperparameters

- num channels - the number of convolutional channels to apply
- learning rate - the learning rate for the optimizer
- batch size - the size of batches the use during training
- epochs - the number of epochs to train for

See 'hyperparameter_tuning.txt' for the output from the grid search. 

It seems to me like a smaller number of channels and a smaller learning rate seem to reliably produce better test loss.

Batch size is more all over the place.

In terms of epochs, the longer they train, the more they seem to overfit - see below:

![](/assets/images/overfitting.png?raw=True "Overfitting")

So, for my final model, I'll use 1 channel, a learning rate of .0001, a batch size of 32, trained for 10 epochs.

## Model

My final model, when evaluated on the holdout evaluation set:

- ROC: 0.75
- Accuracy: 72.5%
- Sensitivity: 82.1%
- Specificity: 56.8%
- PPV: 75.7%
- NPV: 66.0%

![](/assets/images/model_1_cm.png?raw=True "Model 1 Confusion Matrix")

## Analyze Tool

In analyze.ipynb, you can choose any audio clip from the dataset, and it will get loaded in, analyzed by the model, and the estimated probability that it's a deepfake will be displayed, along with the sub-clips that most triggered the deepfake detector.

