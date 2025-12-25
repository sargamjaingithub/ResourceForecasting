import torch
import torch.nn as nn

from app.models.networks.network_interface import NetworkInterface


class DeepConvolutionalClassifierInceptionBranch3NoEncBigBottleneckComposed(NetworkInterface):
    def __init__(self, lag_size: int, prediction_size: int, number_features: int, device: str):
        super(DeepConvolutionalClassifierInceptionBranch3NoEncBigBottleneckComposed, self).__init__(
            lag_size,
            prediction_size,
            number_features,
            device
        )
        self.lag_size = lag_size
        self.number_features = number_features
        self.feature_reduction = nn.Sequential(
            nn.Conv1d(lag_size, lag_size * 2, kernel_size=(number_features,), stride=(3,), padding=0, groups=lag_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size * 2)
        )
        self.inception_1 = nn.Sequential(
            nn.Conv1d(lag_size * 2, lag_size, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size),
            nn.Conv1d(lag_size, lag_size // 2, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 2),
            nn.Flatten(),
            nn.Linear(lag_size // 2, lag_size // 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 8)
        )
        self.upsample_1 = nn.Sequential(
            nn.Conv1d(lag_size // 8, lag_size // 2, kernel_size=(1,), stride=(2,), padding=0)
        )
        self.inception_2 = nn.Sequential(
            nn.Conv1d(lag_size + lag_size // 2, lag_size // 2, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 2),
            nn.Conv1d(lag_size // 2, lag_size // 4, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 4),
            nn.Flatten(),
            nn.Linear(lag_size // 4, lag_size // 16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 16)
        )
        self.upsample_2 = nn.Sequential(
            nn.Conv1d(lag_size // 16, lag_size // 4, kernel_size=(1,), stride=(2,), padding=0)
        )
        self.inception_3 = nn.Sequential(
            nn.Conv1d(lag_size // 2 + lag_size // 4, lag_size // 4, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 4),
            nn.Conv1d(lag_size // 4, lag_size // 8, kernel_size=(1,), stride=(2,), padding=0),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 8),
            nn.Flatten(),
            nn.Linear(lag_size // 8, lag_size // 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(lag_size // 32)
        )
        self.union = nn.Sequential(
            nn.Linear(lag_size // 8 + lag_size // 16 + lag_size // 32, prediction_size),
        )

    def forward(self, x, y=None):
        x = torch.transpose(x, 1, 2)
        x = self.feature_reduction(x)
        x1 = self.inception_1(x)
        x1_upsampled = self.upsample_1(x1.reshape((x.shape[0], self.lag_size // 8, 1)))
        x2 = torch.cat((x1_upsampled, x[:, -self.lag_size:]), 1)
        x2 = self.inception_2(x2)
        x2_upsampled = self.upsample_2(x2.reshape((x.shape[0], self.lag_size // 16, 1)))
        x3 = torch.cat((x2_upsampled, x[:, -(self.lag_size // 2):]), 1)
        x3 = self.inception_3(x3)
        x = torch.cat((x1, x2, x3), 1)
        return self.union(x)

    def predict(self, x):
        return self.forward(x, None)

