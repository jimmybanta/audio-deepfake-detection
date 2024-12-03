import pytest
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
import torchaudio
from data.import_data import pad_tensor, load_file, load_dataset, get_dataloader
from dotenv import load_dotenv

load_dotenv()



@pytest.fixture
def assets_path():
    return os.path.join(os.getenv('HOME_DIR'), 'tests', 'data_tests', 'assets')

def test_pad_tensor():
    tensor = torch.randn(1, 128, 75)
    target_length = 80
    padded_tensor = pad_tensor(tensor, target_length)
    assert padded_tensor.shape[2] % target_length == 0

def test_load_file(assets_path):
    file_path = os.path.join(assets_path, '4.wav')

    target_length = 80
    file_tensors = load_file(file_path, target_length)
    for tensor in file_tensors:
        assert tensor.shape[2] == target_length

def test_load_dataset(assets_path):
    meta_file = os.path.join(assets_path, 'meta.csv')
    sample_rate = 16000

    target_length = 80
    dataset = load_dataset(meta_file, target_length, files_to_load=2)
    assert isinstance(dataset, TensorDataset)
    assert dataset[0][0].shape == torch.Size([1, 128, target_length])
    assert dataset[0][1].shape == torch.Size([2])
    

def test_get_dataloader():
    data = torch.randn(10, 1, 128, 80)
    labels = torch.randint(0, 2, (10, 2))
    dataset = TensorDataset(data, labels)
    dataloader = get_dataloader(dataset, batch_size=2)
    assert isinstance(dataloader, DataLoader)
    for batch_data, batch_labels in dataloader:
        assert batch_data.shape[0] == 2
        assert batch_labels.shape[0] == 2

if __name__ == '__main__':
    pytest.main()