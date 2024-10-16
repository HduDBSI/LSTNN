import torch

from ..base_tsf_runner import BaseTimeSeriesForecastingRunner


class InOutformerRunner(BaseTimeSeriesForecastingRunner):
    """Runner for NewCrossformer: add setup_graph and teacher forcing."""

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.forward_features = cfg["MODEL"].get("FORWARD_FEATURES", None)
        self.target_features = cfg["MODEL"].get("TARGET_FEATURES", None)

    def setup_graph(self, data):
        """The dcrnn official codes act like tensorflow, which create parameters in the first feedforward process."""
        try:
            self.train_iters(1, 0, data)
        except AttributeError:
            pass

    def select_input_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select input features and reshape data to fit the target model.

        Args:
            data (torch.Tensor): input history data, shape [B, L, N, C].

        Returns:
            torch.Tensor: reshaped data
        """

        # select feature using self.forward_features
        if self.forward_features is not None:
            data = data[:, :, :, self.forward_features]
        return data

    def select_target_features(self, data: torch.Tensor) -> torch.Tensor:
        """Select target features and reshape data back to the BasicTS framework

        Args:
            data (torch.Tensor): prediction of the model with arbitrary shape.

        Returns:
            torch.Tensor: reshaped data with shape [B, L, N, C]
        """

        # select feature using self.target_features
        data = data[:, :, :, self.target_features]
        return data

    def forward(self, data: tuple, epoch: int = None, iter_num: int = None, train: bool = True, **kwargs) -> tuple:
        """Feed forward process for train, val, and test. Note that the outputs are NOT re-scaled.

        Args:
            data (tuple): data (future data, history data). [B, L, N, C] for each of them
            epoch (int, optional): epoch number. Defaults to None.
            iter_num (int, optional): iteration number. Defaults to None.
            train (bool, optional): if in the training process. Defaults to True.

        Returns:
            tuple: (prediction, real_value)
        """

        # preprocess
        future_data, history_data, tpl_matrix = data
        history_data = self.to_running_device(history_data)      # B, L, N, C
        future_data = self.to_running_device(future_data)        # B, L, N, C
        tpl_matrix = self.to_running_device(tpl_matrix)          # B, L, N, N

        batch_size, length, num_nodes, _ = future_data.shape

        history_data = self.select_input_features(history_data)
        _future_data = self.select_input_features(future_data)

        # 把B, L, N, 1 -> B, L, N
        history_data = torch.squeeze(history_data, -1)

        # feed forward
        prediction_data = self.model(history_data, tpl_matrix)

        # post process
        prediction = self.select_target_features(prediction_data)

        real_value = _future_data
        return prediction, real_value
