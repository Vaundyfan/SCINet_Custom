import os
import math
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from metrics.Finantial_metics import MSE, MAE
from experiments.exp_basic import Exp_Basic
from data_process.financial_dataloader import DataLoaderH
from utils.tools import EarlyStopping, adjust_learning_rate, save_model, load_model
from metrics.ETTh_metrics import metric
from utils.math_utils import smooth_l1_loss
from models.SCINet import SCINet
from models.SCINet_decompose import SCINet_decompose

class Exp_financial(Exp_Basic):
    def __init__(self, args):
        super(Exp_financial, self).__init__(args)
        self.criterion = smooth_l1_loss if self.args.L1Loss else nn.MSELoss(size_average=False).cuda()
        self.evaluateL2 = nn.MSELoss(size_average=False).cuda()
        self.evaluateL1 = nn.L1Loss(size_average=False).cuda()
        self.writer = SummaryWriter('.exp/run_financial/{}'.format(args.model_name))
    
    def _build_model(self):
        # Adjusting input dimension for your dataset
        #if self.args.dataset_name == 'MK2000_with_macro_and_volatility':
        self.input_dim = 10  # Adjust according to the number of features in your dataset

        model_class = SCINet_decompose if self.args.decompose else SCINet
        model = model_class(
            output_len=self.args.horizon,
            input_len=self.args.window_size,
            input_dim=self.input_dim,
            hid_size=self.args.hidden_size,
            num_stacks=self.args.stacks,
            num_levels=self.args.levels,
            num_decoder_layer=self.args.num_decoder_layer,
            concat_len=self.args.concat_len,
            groups=self.args.groups,
            kernel=self.args.kernel,
            dropout=self.args.dropout,
            single_step_output_One=self.args.single_step_output_One,
            positionalE=self.args.positionalEcoding,
            modified=True,
            RIN=self.args.RIN
        )
        print(model)
        return model
    
    def _get_data(self):
        # Ensure correct path for your CSV file
        if self.args.dataset_name == 'MK2000_with_macro_and_volatility':
            self.args.data = '/content/drive/MyDrive/SCINet/datasets/financial/MK2000_with_macro_and_volatility.csv'
        normalize_type = 4 if self.args.long_term_forecast else self.args.normalize
        return DataLoaderH(self.args.data, 0.7, 0.1, self.args.horizon, self.args.window_size, normalize_type)

    def _select_optimizer(self):
        return optim.Adam(params=self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), weight_decay=self.args.weight_decay)

    def train(self):
        best_val = float("inf")
        optim = self._select_optimizer()
        data = self._get_data()
        X, Y = data.train[0], data.train[1]
        save_path = os.path.join(self.args.save_path, self.args.model_name)
        os.makedirs(save_path, exist_ok=True)

        epoch_start = 0
        if self.args.resume:
            self.model, lr, epoch_start = load_model(self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)

        for epoch in range(epoch_start, self.args.epochs):
            epoch_start_time = time.time()
            self.model.train()
            total_loss, final_loss, min_loss, n_samples = 0, 0, 0, 0

            lr = adjust_learning_rate(optim, epoch, self.args)

            for tx, ty in data.get_batches(X, Y, self.args.batch_size, True):
                self.model.zero_grad()
                forecast = self.model(tx) if self.args.stacks == 1 else self.model(tx)[0]
                scale = data.scale.expand(forecast.size(0), self.args.horizon, data.m)
                bias = data.bias.expand(forecast.size(0), self.args.horizon, data.m)

                if self.args.single_step:
                    ty_last = ty[:, -1, :]
                    scale_last, bias_last = data.scale.expand(forecast.size(0), data.m), data.bias.expand(forecast.size(0), data.m)
                    loss_f = self.criterion(forecast[:, -1] * scale_last + bias_last, ty_last * scale_last + bias_last)
                else:
                    loss_f = self.criterion(forecast * scale + bias, ty * scale + bias)

                loss = loss_f
                loss.backward()
                optim.step()
                total_loss += loss.item()
                final_loss += loss_f.item()
                n_samples += (forecast.size(0) * data.m)

            print(f'| End of epoch {epoch+1} | time: {(time.time() - epoch_start_time):.2f}s | '
                  f'train_loss {total_loss / n_samples:.4f}', flush=True)

            val_metrics = self.validate(data, data.valid[0], data.valid[1])
            val_loss, _, _, _, _ = val_metrics
            if val_loss < best_val:
                save_model(epoch, lr, self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)
                print('Saved new best model!')
                best_val = val_loss

    def validate(self, data, X, Y, evaluate=False):
        self.model.eval()
        total_loss, total_loss_l1, n_samples = 0, 0, 0
        forecast_set, target_set = [], []

        if evaluate:
            save_path = os.path.join(self.args.save_path, self.args.model_name)
            self.model = load_model(self.model, save_path, model_name=self.args.dataset_name, horizon=self.args.horizon)[0]

        for tx, ty in data.get_batches(X, Y, self.args.batch_size, False):
            with torch.no_grad():
                forecast = self.model(tx) if self.args.stacks == 1 else self.model(tx)[0]
                scale = data.scale.expand(forecast.size(0), data.m)
                bias = data.bias.expand(forecast.size(0), data.m)
                output = forecast[:, -1, :] * scale + bias
                true = ty[:, -1, :] * scale + bias

                total_loss += self.evaluateL2(output, true).item()
                total_loss_l1 += self.evaluateL1(output, true).item()
                n_samples += (output.size(0) * data.m)
                forecast_set.append(forecast)
                target_set.append(ty)

        mse = MSE(torch.cat(forecast_set).cpu().numpy(), torch.cat(target_set).cpu().numpy())
        mae = MAE(torch.cat(forecast_set).cpu().numpy(), torch.cat(target_set).cpu().numpy())
        rse = math.sqrt(total_loss / n_samples) / data.rse
        rae = total_loss_l1 / n_samples / data.rae
        correlation = np.corrcoef(output.cpu().numpy().flatten(), true.cpu().numpy().flatten())[0, 1]

        print(f'Validation - mse: {mse:.4f}, mae: {mae:.4f}, rse: {rse:.4f}, rae: {rae:.4f}, corr: {correlation:.4f}')
        return rse, rae, correlation, mse, mae

