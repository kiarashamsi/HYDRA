from typing import List
import torch.nn as nn
from torch_geometric.nn.models import MLP
from torch.nn import GRU, LSTM, RNN, Sequential, Linear, ReLU
import torch


class MLP_RNN_Wrapper(nn.Module):
    r"""
    # RNNs (include LSTM, GRU) and MLP wrapper
    Changed: input dimension from (seq_len/batch_size,hidden_dim) -> (hidden_dim), since we assume seq_len/batch_size = 1
    Args:
        - model (nn.Module): models from torch or PyG
    
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input: torch.Tensor, hx=None):
        input = input.unsqueeze(0)
        output = self.model(input, hx)
        if len(output) > 1:
            output, _ = output
        output = output.squeeze(0)
        return output

class MultiHeadDecoder(nn.Module):
    def __init__(self,
            in_channels: int,
            out_channels: int,
            heads_num : int,
            hids_size: List[int] = [32],
            nlayers: List[int] = [2],
            models_seq: List[str] = ['GRU', 'FC'],
            bias: List[bool] = [True],
            drop_out: List[float] = [0.1]
            ):
        super().__init__()
        r"""
        Multi-head Decoder.

        Each head is a flexible version of decoder implemented belows
        """
        self.decoders = torch.nn.ModuleList([
            Decoder(in_channels= in_channels, out_channels= out_channels, hids_size=hids_size, nlayers = nlayers,models_seq=models_seq,
                    bias= bias, drop_out= drop_out)
            for _ in range(heads_num)
        ])

    def forward(self,x):
        outputs = []
        for decoder in self.decoders:
            outputs.append(decoder(x).squeeze(0))
        
        return torch.cat(outputs,dim=0)


class Decoder(nn.Module):
    r"""
    # Flexible version of decoder.

    Decoder can be one of following models
        - MLP
        - RNN
        - LSTM
        - GRU
        - Fully connected layer
    or a combination of them.

    Args:
        in_channels(int) : dimension of input
        out_channels:(int) : dimension of output
        hids_size (List[int]) : hidden dimension of each models
        nlayers (List[int]) : number of layers of each models
        models_seq (List[str]) : sequence of models used in decoder (order matter)
        bias (List[bool]) : list of boolean to define whether to use bias for each model
        drop_out (List[float]) : list of float to define drop out value for each model

    *Last layer must be MLP or Fully connected module*

    Example:
        decoder = Decoder(8,1,hids_size=[32,64,128],nlayers=[2,3,4],models_seq=["LSTM","GRU","MLP"],drop_out=[0.3,0.5,0.5])
        *This create decoder in order LSTM->GRU->MLP. The decoder has architecture as follow:*
        (decoder): Sequential(

            (0): MLP_RNN_Wrapper(
            (model): LSTM(in_channel=8, hidden_size = 32, num_layers=2, dropout=0.3)
            )

            (1): MLP_RNN_Wrapper(
            (model): GRU(in_channel = 32, hidden_size = 64, num_layers=3, dropout=0.5)
            )

            (2): MLP_RNN_Wrapper(
            (model): MLP(in_channel = 64, hidden_size = 128, num_layers = 4,out_channel =  1)
        )


    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            hids_size: List[int] = [32],
            nlayers: List[int] = [2],
            models_seq: List[str] = ['GRU', 'FC'],
            bias: List[bool] = [True],
            drop_out: List[float] = [0.1]
    ):

        nums_module = len(models_seq)
        assert nums_module >= 1, "Decoder must have at least one model"
        assert len(drop_out) == nums_module or len(
            drop_out) == 1, "Drop out should be define for each models or define once"
        assert len(nlayers) == nums_module or len(
            nlayers) == 1, "Number of layers should be define for each models or define once"
        assert len(hids_size) == nums_module or len(
            hids_size) == 1, "Number of hiddens should be define for each models or define once"
        assert len(bias) == nums_module or len(
            bias) == 1, "Number of bias should be define for each models or define once"
        assert len(drop_out) == nums_module or len(
            drop_out) == 1, "Number of drop out values should be define for each models or define once"
        assert models_seq[-1] == "MLP" or models_seq[-1] == "FC", "Last layer must be MLP or Fully Connect layer"
        super().__init__()
        hids_size = self._repeat(hids_size, nums_module)
        nlayers = self._repeat(nlayers, nums_module)
        drop_out = self._repeat(drop_out, nums_module)
        bias = self._repeat(bias, nums_module)

        self.decoder = Sequential()
        if nums_module == 1:
            self.decoder.append(
                self._init_model(models_seq[0], in_channels, hids_size[0], nlayers[0], drop_out[0], bias[0],
                                 out_channels)
            )
        else:
            self.decoder.append(
                self._init_model(models_seq[0], in_channels, hids_size[0], nlayers[0], drop_out[0], bias[0])
            )

            for i in range(1, nums_module - 1):
                self.decoder.append(
                    self._init_model(models_seq[1], hids_size[i - 1], hids_size[i], nlayers[i], drop_out[i], bias[i])
                )

            self.decoder.append(
                self._init_model(models_seq[-1], hids_size[-2], hids_size[-1], nlayers[-1], drop_out[-1], bias[-1],
                                 out_channels)
            )

    def _init_model(self,
                    model_name: str,
                    in_channel: int,
                    hid_size: int,
                    num_layers: int,
                    drop_out: float,
                    bias: bool,
                    out_channel: int = None) -> nn.Module:

        model = self._model_resolver(model_name)
        if model_name in ['LSTM', 'GRU', 'RNN']:
            return MLP_RNN_Wrapper(model(
                input_size=in_channel, hidden_size=hid_size, num_layers=num_layers, bias=bias, dropout=drop_out
            ))
        elif model_name == 'FC':
            return model(
                in_features=in_channel, out_features=out_channel, bias=bias
            )
        elif model_name == "MLP":
            return MLP_RNN_Wrapper(model(
                in_channels=in_channel, hidden_channels=hid_size, out_channels=out_channel, num_layers=num_layers,
                dropout=drop_out, bias=bias, norm=None
            ))

    def _repeat(self, list: List, times: int):
        if len(list) != times:
            list = list * times
        return list

    def _model_resolver(self, models_name: str = "LSTM"):
        if models_name == "LSTM":
            return LSTM
        elif models_name == "GRU":
            return GRU
        elif models_name == "RNN":
            return RNN
        elif models_name == "MLP":
            return MLP
        elif models_name == "FC":
            return Linear
        else:
            raise Exception(f"Unsupport decoder {models_name}")

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        # out = self.decoder(x)
        return self.decoder(x)


# Running Example

# if __name__ == "__main__":
#     #############
#     in_tensor = torch.randn(8)
#     print(in_tensor)
#     decoder = Decoder(8, 1)
#     # decoder = Decoder(8,1,hids_size=[32],nlayers=[2],models_seq=["MLP"],drop_out=[0.3])
#
#     out_tensor = decoder(in_tensor)
#
#     print(out_tensor.shape)
