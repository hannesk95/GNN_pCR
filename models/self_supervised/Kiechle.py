import sys
sys.path.append("/home/johannes/Data/SSD_2.0TB/GNN_pCR/")

import monai
import torch
import torch.nn as nn
from torch_geometric.nn.conv import SAGEConv
from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.typing import Adj, OptPairTensor, Size
from utils.graph_utils import make_directed_complete_forward_graph
from torch_geometric.data import Batch

class WeightedSAGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        aggr: Optional[Union[str, List[str], Aggregation]] = "mean",
        normalize: bool = False,
        root_weight: bool = True,
        project: bool = False,
        bias: bool = True,
        **kwargs,
    ):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.root_weight = root_weight
        self.project = project

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        if aggr == 'lstm':
            kwargs.setdefault('aggr_kwargs', {})
            kwargs['aggr_kwargs'].setdefault('in_channels', in_channels[0])
            kwargs['aggr_kwargs'].setdefault('out_channels', in_channels[0])

        super().__init__(aggr, **kwargs)

        if self.project:
            if in_channels[0] <= 0:
                raise ValueError(f"'{self.__class__.__name__}' does not "
                                 f"support lazy initialization with "
                                 f"`project=True`")
            self.lin = Linear(in_channels[0], in_channels[0], bias=True)

        if isinstance(self.aggr_module, MultiAggregation):
            aggr_out_channels = self.aggr_module.get_out_channels(
                in_channels[0])
        else:
            aggr_out_channels = in_channels[0]

        self.lin_l = Linear(aggr_out_channels, out_channels, bias=bias)
        if self.root_weight:
            self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        if self.project:
            self.lin.reset_parameters()
        self.lin_l.reset_parameters()
        if self.root_weight:
            self.lin_r.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        size: Size = None,
    ) -> Tensor:

        if isinstance(x, Tensor):
            x = (x, x)

        if self.project and hasattr(self, 'lin'):
            x = (self.lin(x[0]).relu(), x[1])

        # propagate_type: (x: OptPairTensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)

        x_r = x[1]
        if self.root_weight and x_r is not None:
            out = out + self.lin_r(x_r)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor, edge_weight: Optional[Tensor]) -> Tensor:
        if edge_weight is None:
            return x_j
        return edge_weight.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, aggr={self.aggr})')

class TemporalGNN(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(TemporalGNN, self).__init__()

        self.conv1 = SAGEConv(in_channels, out_channels, aggr="mean")
        self.conv2 = SAGEConv(out_channels, out_channels, aggr="mean")
        self.relu = torch.nn.ReLU()
        self.bn = BatchNorm(out_channels)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = self.relu(x)        
        x = self.conv2(x, edge_index)
        
        return x
    
class TemporalWeightGNN(torch.nn.Module):

    def __init__(self, in_channels, out_channels):
        super(TemporalWeightGNN, self).__init__()

        self.conv1 = WeightedSAGEConv(in_channels, out_channels, aggr="mean")
        self.conv2 = WeightedSAGEConv(out_channels, out_channels, aggr="mean")
        self.relu = torch.nn.ReLU()
        self.bn = BatchNorm(out_channels)

    def forward(self, x, edge_index, edge_weight=None):

        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.bn(x)
        x = self.relu(x)        
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        
        return x

class LinearMLP(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearMLP, self).__init__()

        self.linear1 = torch.nn.Linear(in_channels, out_channels)        
        self.linear2 = torch.nn.Linear(out_channels, out_channels)
        self.relu = torch.nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
    
class ResNet18EncoderKiechle(torch.nn.Module):
    def __init__(self, timepoints, use_gnn=True, method=""):
        super(ResNet18EncoderKiechle, self).__init__()
        
        self.use_gnn = use_gnn
        self.timepoints = timepoints
        self.dist_method = "graph_edges" if "dist_graph" in method else "feature_append"

        model = monai.networks.nets.resnet18(spatial_dims=3, n_input_channels=3, num_classes=2)
        encoder_layers = list(model.children())[:-1]
        encoder_layers.append(torch.nn.Flatten(start_dim=1, end_dim=-1))                
        self.encoder = nn.Sequential(*encoder_layers)

        if use_gnn:
            if method == "kiechle_dist_graph":
                self.gnn_projector = TemporalWeightGNN(in_channels=512, out_channels=128)
            else:
                self.gnn_projector = TemporalGNN(in_channels=512, out_channels=128)
        else:
            self.mlp_projector = LinearMLP(in_channels=512, out_channels=128)
        

    def forward(self, images, time_distances=None):
        """
        images:   (B, T, C, D, H, W)
        """

        if time_distances is not None:
            if self.dist_method == "graph_edges":
                time_distances = time_distances.squeeze()
                time_distances = time_distances.flatten().cuda()
            else:
                pass # do nothing as time distances are already in the correct shape for appending to feature vectors

        B, T, C, D, H, W = images.shape
   
        images = images.view(B * T, C, D, H, W)
        features = self.encoder(images)  # (B*T, 512)

        features = features.view(B, T, -1)  # (B, T, 512)

        features_original = features[:, :self.timepoints, :]  # (B, timepoints, 512)
        features_transformed = features[:, self.timepoints:, :]  # (B, timepoints, 512)

        graph_data_original = make_directed_complete_forward_graph(features_original, batch_size=B)
        graph_data_original = graph_data_original.to(features.device)
        # graph_data_original = graph_data_original.sort(sort_by_row=False)

        graph_data_transformed = make_directed_complete_forward_graph(features_transformed, batch_size=B)        
        graph_data_transformed = graph_data_transformed.to(features.device)
        # graph_data_transformed = graph_data_transformed.sort(sort_by_row=False)

        if self.use_gnn:
            if self.dist_method == "graph_edges":
                latents_original = self.gnn_projector(graph_data_original.x, graph_data_original.edge_index, edge_weight=time_distances)  # (B*timepoints, 128)
            else:
                latents_original = self.gnn_projector(graph_data_original.x, graph_data_original.edge_index)  # (B*timepoints, 128)
        else:
            latents_original = self.mlp_projector(graph_data_original.x)  # (B*timepoints, 128)
        latents_original = latents_original.view(B, -1, latents_original.size(-1))

        if self.dist_method == "feature_append":
            latents_original = torch.cat([latents_original, time_distances], dim=-1)
            
        latents_original = latents_original.flatten(start_dim=1) 
        
        if self.use_gnn:
            if self.dist_method == "graph_edges":
                latents_transformed = self.gnn_projector(graph_data_transformed.x, graph_data_transformed.edge_index, edge_weight=time_distances)  # (B*timepoints, 128)
            else:
                latents_transformed = self.gnn_projector(graph_data_transformed.x, graph_data_transformed.edge_index)  # (B*timepoints, 128)
        else:
            latents_transformed = self.mlp_projector(graph_data_transformed.x)  # (B*timepoints, 128)
        latents_transformed = latents_transformed.view(B, -1, latents_transformed.size(-1))

        if self.dist_method == "feature_append":
            latents_transformed = torch.cat([latents_transformed, time_distances], dim=-1)

        latents_transformed = latents_transformed.flatten(start_dim=1) 
        
        latents = torch.concat([latents_original, latents_transformed], dim=0)
        
        return latents
    

