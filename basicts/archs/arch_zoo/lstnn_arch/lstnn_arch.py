from math import ceil

import torch
from torch import nn
from basicts.archs.arch_zoo.lstnn_arch.patch_emb import PatchEncoder
from basicts.archs.arch_zoo.lstnn_arch.downsamp_emb import DownsampEncoder


class MultiscaleMLP(nn.Module):
    #  ,**model_args
    def __init__(self, **model_args):
        super(MultiscaleMLP, self).__init__()
        self.node_size = model_args["node_size"]
        self.input_len = model_args["input_len"]
        self.input_dim = model_args["input_dim"]
        self.output_len = model_args["output_len"]
        self.patch_len = model_args["patch_len"]
        self.stride = model_args["stride"]
        self.td_size = model_args["td_size"]
        self.dw_size = model_args["dw_size"]
        self.d_td = model_args["d_td"]
        self.d_dw = model_args["d_dw"]
        self.d_d = model_args["d_d"]
        self.d_spa = model_args["d_spa"]

        self.if_time_in_day = model_args["if_time_in_day"]
        self.if_day_in_week = model_args["if_day_in_week"]
        self.if_spatial = model_args["if_spatial"]
        self.num_layer = model_args["num_layer"]

        # temporal embeddings
        if self.if_time_in_day:
            self.td_codebook = nn.Parameter(torch.empty(self.td_size, self.d_td))
            nn.init.xavier_uniform_(self.td_codebook)

        if self.if_day_in_week:
            self.dw_codebook = nn.Parameter(torch.empty(self.dw_size, self.d_dw))
            nn.init.xavier_uniform_(self.dw_codebook)

        # spatial embeddings
        if self.if_spatial:
            self.spa_codebook = nn.Parameter(torch.empty(self.node_size, self.d_spa))
            nn.init.xavier_uniform_(self.spa_codebook)

        # Encoder
        self.patch_encoder = PatchEncoder(self.td_size, self.td_codebook, self.dw_codebook, self.spa_codebook, self.if_time_in_day, self.if_day_in_week, self.if_spatial,
                                          self.input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa, self.output_len, self.num_layer)

        self.downsamp_encoder = DownsampEncoder(self.td_size, self.td_codebook, self.dw_codebook, self.spa_codebook, self.if_time_in_day, self.if_day_in_week, self.if_spatial,
                                          self.input_dim, self.patch_len, self.stride, self.d_d, self.d_td, self.d_dw, self.d_spa, self.output_len, self.num_layer)

        # Residual
        self.residual = nn.Conv2d(in_channels=self.input_len, out_channels=self.output_len, kernel_size=(1, 1), bias=True)

    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor, batch_seen: int, epoch: int, train: bool, **kwargs) -> torch.Tensor:
        """
        Args:   history_data (torch.Tensor): history data with shape [B, L, N, C]
        Returns:    torch.Tensor: prediction wit shape [B, L, N, C]
        """
        # prepare data
        input_data = history_data[..., range(self.input_dim)]

        # patching
        in_len_add = ceil(1.0 * self.input_len / self.stride) * self.stride - self.input_len
        if not in_len_add:
            input_data = torch.cat((input_data[:, -1:, :, :].expand(-1, in_len_add, -1, -1), input_data), dim=1)

        # 下采样patch
        downsamp_input = [input_data[:, i::self.stride, :, :] for i in range(self.stride)]
        downsamp_input = torch.stack(downsamp_input, dim=1)

        # 分段patch
        patch_input = input_data.unfold(dimension=1, size=self.patch_len, step=self.patch_len).permute(0, 1, 4, 2, 3)  # B P L N C
        _, p_num, _, _, _ = patch_input.shape

        patch_predict = self.patch_encoder(patch_input)
        downsamp_predict = self.downsamp_encoder(downsamp_input)

        # Residual
        output = patch_predict + downsamp_predict + self.residual(input_data)
        # output = patch_predict

        return output
