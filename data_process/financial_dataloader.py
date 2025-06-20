from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import numpy as np
import pandas as pd

def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))

class DataLoaderH(object):
    def __init__(self, file_name, train, valid, horizon, window, normalize=2):
        self.P = window
        self.h = horizon
        # Load data using pandas and select only numeric columns
        try:
            df = pd.read_csv('/content/drive/MyDrive/SCINet/datasets/financial/MK2000_with_macro_and_volatility.csv')
            df = df.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric values to NaN
            df = df.dropna(axis=1, how='any')  # Drop columns with any NaN values

            self.rawdat = df.values  # Convert cleaned DataFrame to numpy array
        except Exception as e:
            raise ValueError(f"Error loading file: {e}")

        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.normalize = normalize
        self.scale = np.ones(self.m)
        self.bias = np.zeros(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)

        # Convert scale and bias to tensors
        self.scale = torch.from_numpy(self.scale).float()
        self.bias = torch.from_numpy(self.bias).float()
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.h, self.m)

        # Move to GPU if available
        if torch.cuda.is_available():
            self.scale = self.scale.cuda()
            self.scale = Variable(self.scale)
            self.bias = self.bias.cuda()
            self.bias = Variable(self.bias)

        tmp = tmp[:, -1, :].squeeze()
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    # Rest of the class methods remain the same...


    def _normalized(self, normalize):
        if normalize == 0:
            self.dat = self.rawdat
        elif normalize == 1:
            self.dat = self.rawdat / np.max(self.rawdat)
        elif normalize == 2:
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, i]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))
        elif normalize == 3:
            for i in range(self.m):
                self.scale[i] = np.std(self.rawdat[:, i])
                self.bias[i] = np.mean(self.rawdat[:, i])
                self.dat[:, i] = (self.rawdat[:, i] - self.bias[i]) / self.scale[i]
        elif normalize == 4:
            for i in range(self.m):
                self.scale[i] = np.std(self.rawdat[:int(self.dat.shape[0] * 0.7), i])
                self.bias[i] = np.mean(self.rawdat[:int(self.dat.shape[0] * 0.7), i])
                self.dat[:, i] = (self.rawdat[:, i] - self.bias[i]) / self.scale[i]

    def _split(self, train, valid, test):
        train_set = range(self.P + self.h - 1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        self.train = self._batchify(train_set, self.h)
        self.valid = self._batchify(valid_set, self.h)
        self.test = self._batchify(test_set, self.h)

    def _batchify(self, idx_set, horizon):
        n = len(idx_set)
        X = torch.zeros((n, self.P, self.m))
        Y = torch.zeros((n, self.h, self.m))
        for i in range(n):
            end = idx_set[i] - self.h + 1
            start = end - self.P
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :, :] = torch.from_numpy(self.dat[end:(idx_set[i]+1), :])
        return [X, Y]

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while start_idx < length:
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt]
            Y = targets[excerpt]
            if torch.cuda.is_available():
                X = X.cuda()
                Y = Y.cuda()
            yield Variable(X), Variable(Y)
            start_idx += batch_size


