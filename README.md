# Audio Deepfake Detection


## Data Collection

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




## To Do:

- [x] Collect data
- [x] split data
- [ ] Data pre-processing
- [ ] Build CNN
- [ ] CNN Training
