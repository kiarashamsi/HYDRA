from typing import Iterator
import torch
from torch.nn.parameter import Parameter
from torch_geometric.nn.models import MLP
from torch_geometric.utils import negative_sampling


EPS = 1e-15
MAX_LOGVAR = 10

class MLP_Decoder(torch.nn.Module):
    r"""
    MLP decoder for link prediction with MLP is implemened by PyG team

    mlp_args: arguments to init MLP module from PyG
        channel_list (List[int] or int, optional): List of input, intermediate
            and output channels such that :obj:`len(channel_list) - 1` denotes
            the number of layers of the MLP (default: :obj:`None`)
        in_channels (int, optional): Size of each input sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        hidden_channels (int, optional): Size of each hidden sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        out_channels (int, optional): Size of each output sample.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        num_layers (int, optional): The number of layers.
            Will override :attr:`channel_list`. (default: :obj:`None`)
        dropout (float or List[float], optional): Dropout probability of each
            hidden embedding. If a list is provided, sets the dropout value per
            layer. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        norm (str or Callable, optional): The normalization function to
            use. (default: :obj:`"batch_norm"`)
        norm_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective normalization function defined by :obj:`norm`.
            (default: :obj:`None`)
        plain_last (bool, optional): If set to :obj:`False`, will apply
            non-linearity, batch normalization and dropout to the last layer as
            well. (default: :obj:`True`)
        bias (bool or List[bool], optional): If set to :obj:`False`, the module
            will not learn additive biases. If a list is provided, sets the
            bias per layer. (default: :obj:`True`)
        
    """
    def __init__(self,sampling_times,**mlp_args):
        self.sampling_times = sampling_times
        self.mlp = MLP(**mlp_args)
        self.negative_sampling = negative_sampling

    def decoder(self, z, edge_index):
        value = (z[edge_index[0]] * z[edge_index[1]])
        return (self.mlp(value)).sigmoid()
    
    def forward(self,z,pos_edge_index_list, neg_edge_index_list=None):
        poss_loss = torch.tensor(0,requires_grad=True)      
        for pos_edge in pos_edge_index_list:
            poss_loss += self.decoder(z,pos_edge)
        poss_loss =-torch.log(poss_loss/len(pos_edge_index_list))


        if neg_edge_index == None:
            print("INFO: generating negative samples...")
            neg_edge_index = negative_sampling(pos_edge_index_list,
                                                num_neg_samples=pos_edge_index_list.size(1) * self.sampling_times)
            poss_loss = poss_loss/len(pos_edge_index_list)

        neg_loss = torch.tensor(0,requires_grad=True)      
        for neg_edge in neg_edge_index_list:
            neg_loss += self.decoder(z,neg_edge)
        neg_loss =-torch.log(1- poss_loss/len(pos_edge_index_list))

        return poss_loss + neg_loss
    
    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self.mlp.parameters()
        



