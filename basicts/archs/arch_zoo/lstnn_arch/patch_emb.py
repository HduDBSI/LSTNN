from math import ceil

import torch
from einops import rearrange
from torch import nn
from basicts.archs.arch_zoo.lstnn_arch.mlp import MultiLayerPerceptron


class PatchEncoder(nn.Module):
    def __init__(self, td_size, td_codebook, dw_codebook, spa_codebook, if_time_in_day, if_day_in_week, if_spatial,
                 input_dim, patch_len, stride, d_d, d_td, d_dw, d_spa, output_len, num_layer):
        super(PatchEncoder, self).__init__()
        self.td_codebook = td_codebook
        self.dw_codebook = dw_codebook
        self.spa_codebook = spa_codebook
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.if_spatial = if_spatial
        self.output_len = output_len
        self.td_size = td_size
        self.stride = stride

        self.data_embedding_layer = nn.Conv2d(in_channels=input_dim*patch_len, out_channels=d_d, kernel_size=(1, 1), bias=True)
        self.hidden_dim = d_d + d_dw*int(self.if_day_in_week)*2 + d_td*int(self.if_time_in_day)*2

        self.temporal_encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim+d_spa*int(self.if_spatial), self.hidden_dim+d_spa*int(self.if_spatial)) for _ in range(num_layer)]) # +d_spa*int(self.if_spatial)

        self.spatial_encoder = nn.Sequential(
            *[MultiLayerPerceptron(d_d+d_spa*int(self.if_spatial), d_d+d_spa*int(self.if_spatial)) for _ in range(num_layer)])

        self.data_encoder = nn.Sequential(
            *[MultiLayerPerceptron(d_d, d_d) for _ in range(num_layer)]
        )
        self.projection1 = nn.Conv2d(in_channels=(self.hidden_dim+d_spa*int(self.if_spatial))*self.stride+d_td+d_dw, out_channels=output_len, kernel_size=(1, 1), bias=True)

        # self.projection1 = nn.Conv2d(
        #     in_channels=(self.hidden_dim) * self.stride,
        #     out_channels=output_len, kernel_size=(1, 1), bias=True)

    def forward(self, patch_input):
        # B P L N C
        batch_size, num, _, _, _ = patch_input.shape

        # Temporal Embedding
        if self.if_day_in_week:
            day_in_week_data = patch_input[..., 2]  # B P L N
            day_in_week_start_emb = self.dw_codebook[(day_in_week_data[:, :, 0, :]).type(torch.LongTensor)]  # B P N D
            day_in_week_end_emb = self.dw_codebook[(day_in_week_data[:, :, -1, :]).type(torch.LongTensor)]  # B P N D
            future_day_in_week_emb = day_in_week_end_emb[:, -1, :, :].permute(0, 2, 1).unsqueeze(-1)
        else:
            day_in_week_start_emb, day_in_week_end_emb, future_day_in_week_emb = None, None, None

        if self.if_time_in_day:
            time_in_day_data = patch_input[..., 1]  # B P L N
            time_in_day_start_emb = self.td_codebook[(time_in_day_data[:, :, 0, :] * self.td_size).type(torch.LongTensor)]  # 查询每一个Patch的第一个（当前时间点）的time-day-index 0-287  B P N D
            time_in_day_end_emb = self.td_codebook[(time_in_day_data[:, :, -1, :] * self.td_size).type(torch.LongTensor)]  # 查询每一个Patch的最后一个（当前时间点）的time-day-index 0-287  B P N D
            # 查询未来数据的最后一个时间点的time-day-index B D N 1
            future_time_in_day_emb = self.td_codebook[((time_in_day_data[:, -1, -1, :] * self.td_size + self.output_len) % self.td_size).type(torch.LongTensor)].permute(0, 2, 1).unsqueeze(-1)
        else:
            time_in_day_start_emb, time_in_day_end_emb, future_time_in_day_emb = None, None, None

        # Spatial Embedding
        if self.if_spatial:
            spatial_emb = self.spa_codebook.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(1).expand(-1, num, -1, -1)  # B P N D
        else:
            spatial_emb = None

        # time series embedding
        data_emb = self.data_embedding_layer(torch.concat((patch_input[..., 0], patch_input[..., 1], patch_input[..., 2]), dim=2).permute(0, 2, 1, 3)).permute(0, 2, 3, 1)  # B P N d_d
        data_emb = self.data_encoder(data_emb.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # spatial encoding
        hidden = torch.concat((data_emb, spatial_emb), dim=-1).permute(0, 3, 1, 2)
        hidden = self.spatial_encoder(hidden).permute(0, 2, 3, 1)  # B D P N

        # temporal encoding
        hidden = torch.concat(
            (time_in_day_start_emb, day_in_week_start_emb, hidden, time_in_day_end_emb, day_in_week_end_emb),
            dim=-1).permute(0, 3, 1, 2)  # B D P N
        hidden = self.temporal_encoder(hidden)   # B D P N

        hidden = rearrange(hidden, 'B D P N -> B (D P) N').unsqueeze(-1)
        hidden = torch.concat((hidden, future_time_in_day_emb, future_day_in_week_emb), dim=1)
        predict = self.projection1(hidden)  # B T P N

        return predict
