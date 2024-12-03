''' Contains functions for importing data from file into pytorch. '''

import os
import pandas as pd
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchaudio

load_dotenv()

DATA_DIR = os.getenv('DATA_DIR')

# setup the Mel Spectrogram transform
transform = torchaudio.transforms.MelSpectrogram(16000)

def pad_tensor(tensor, target_length):
    '''
    Given a tensor and a target length, pads the tensor so that its length a multiple of the target length -
    so that it can be split into equal parts.

    Parameters
    ----------
    tensor : torch.tensor
        The tensor to pad
    target_length : int
        The target length to pad the tensor to

    Returns
    -------
    torch.tensor
        The padded tensor
    '''
    
    _, _, length = tensor.shape
    if length % target_length != 0:
        # Calculate padding needed
        padding_needed = target_length - (length % target_length)
        # Pad the tensor
        tensor = torch.nn.functional.pad(tensor, (0, padding_needed))
    return tensor

def load_file(file, target_length=80):
    '''
    Given a file path, loads the file, creates a mel spectrogram, pads it to be cleanly divisible by target_length,
    and splits it into chunks of target_length.

    Parameters
    ----------
    file : str
        The file path to load
    target_length : int
        The target length to split the tensor into

    Returns
    -------
    tuple of torch.tensor
        A tuple of tensors, each of length target_length
    '''

    # load wav file
    waveform, _ = torchaudio.load(file, normalize=True)

    # create mel spectrogram
    mel_specgram = transform(waveform)

    # pad tensor so it's cleanly divisible by target_length
    padded_tensor = pad_tensor(mel_specgram, target_length)

    # return the tensor, split into target_length chunks
    return padded_tensor.split(target_length, dim=2)


def load_dataset(meta_file, 
                 target_length=80, 
                 batch_size=32,
                 files_to_load='all'):
    '''
    Given a meta file, loads in the dataset.

    Parameters
    ----------
    meta_file : str
        The meta file to load
    target_length : int
        The target length to split the tensor into
    batch_size : int
        The batch size to use for the DataLoader
    files_to_load : int or 'all'
        The number of files to load. If 'all', all files are loaded.

    Returns
    -------
    tuple of torch.tensor
        A tuple of tensors - the positives and negatives
    '''

    positives = []
    negatives = []

    # Load the meta file
    meta = pd.read_csv(meta_file)

    # iterate through the files
    for i, row in enumerate(meta.itertuples()):
        
        # If we're only loading a subset of the files, check if we've loaded enough
        if files_to_load != 'all' and i >= files_to_load:
            break

        # Load the file
        file_tensors = load_file(os.path.join(DATA_DIR, row.file), target_length=target_length)

        # Add the tensors to the appropriate list
        for tensor in file_tensors:
            if row.numeric_label == 1:
                positives.append(tensor)
            else:
                negatives.append(tensor)

    # Create labels
    positive_labels = torch.tensor([[1.0, 0.0]] * len(positives))
    negative_labels = torch.tensor([[0.0, 1.0]] * len(negatives))

    # Combine the inputs
    data = torch.cat((torch.stack(positives), torch.stack(negatives)), dim=0)
    # Combine the labels
    labels = torch.cat((positive_labels, negative_labels), dim=0)

    # Create a dataset
    dataset = TensorDataset(data, labels)

    # Create a dataloader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # return the dataloader
    return dataloader