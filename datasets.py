import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from RawBoost import process_Rawboost_feat


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeat = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeat))[:, :max_len][0]
    return padded_x


def label2num(label):
    if label in {'bonafide', 'bona-fide', 'real', 'genuine'}:
        return 1
    elif label in {'spoof', 'fake'}:
        return 0
    else:
        raise ValueError(f'Unknown label: {label}')


class DatasetTrain(Dataset):
    def __init__(self, data_file, wav_file_add=None, algo=0, cut=64600):
        self.wav_file = data_file
        self.wav_file_add = wav_file_add
        self.algo = algo
        self.cut = cut
        self.file_list = []
        self.file2label = {}
        self._add_file_list(data_file)
        if wav_file_add is not None:
            self._add_file_list(wav_file_add)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        label = self.file2label[file]
        x, sr = librosa.load(file, sr=None)
        x = process_Rawboost_feat(x, sr, self.algo)
        x = pad(x, self.cut)
        x = torch.tensor(x, dtype=torch.float32)
        return x, label

    def _add_file_list(self, wav_file):
        # Line: file,label(,attack,speaker,language,etc)
        df = pd.read_csv(wav_file)
        for i in range(len(df)):
            file = df['file'][i]
            label = df['label'][i]
            self.file_list.append(file)
            self.file2label[file] = label2num(label)


class DatasetTest(Dataset):
    def __init__(self, data_file, data_file_add=None, cut=64600,
                 select_criteria=None):
        self.wav_file = data_file
        self.wav_file_add = data_file_add
        self.cut = cut
        self.file_list = []
        self.file2label = {}
        self._add_file_list(data_file, select_criteria)
        if data_file_add is not None:
            self._add_file_list(data_file_add)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        label = self.file2label[file]
        x, sr = librosa.load(file, sr=None)
        x = pad(x, self.cut)
        x = torch.tensor(x, dtype=torch.float32)
        return x, label, file

    def _add_file_list(self, wav_file, select_criteria=None):
        # Line: file,label(,attack,speaker,language,etc)
        df = pd.read_csv(wav_file)
        if select_criteria is not None:
            df = select_criteria(df)
        for i in range(len(df)):
            file = df['file'][i]
            label = df['label'][i]
            self.file_list.append(file)
            self.file2label[file] = label2num(label)

    def _remove_file_list(self, remove_files):
        if not isinstance(remove_files, set):
            remove_files = set(remove_files)

        file_list = []
        for i in range(len(self.file_list)):
            if self.file_list[i] not in remove_files:
                file_list.append(self.file_list[i])
        self.file_list = file_list
