import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter

class ShiftedSoftplus(torch.nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim_list, activation):
        super().__init__()

        layers = []
        
        activation_layer = None
        if activation == "relu":
            activation_layer = nn.ReLU()
        elif activation == "silu":
            activation_layer = nn.SiLU()
        elif activation == "softplus":
            activation_layer = ShiftedSoftplus()
        else:
            assert activation_layer is None

        hidden_dim_list = [input_dim] + hidden_dim_list
        for input_dim, output_dim in zip(hidden_dim_list[:-1], hidden_dim_list[1:]):
            layers.append(nn.Linear(input_dim, output_dim))
            if activation is not None:
                layers.append(activation_layer)

        self.layers = nn.Sequential(*layers)
        return

    def forward(self, x):
        x = self.layers(x)
        return x


class coGNLayer(nn.Module):
    def __init__(
        self, edge_mlp, node_mlp, global_mlp,
        # edge_attention_mlp_local, edge_attention_mlp_global, node_attention_mlp,
        aggregate_edges_local="sum", aggregate_edges_global="sum", aggregate_nodes="sum",
        return_updated_edges=True, return_updated_nodes=True, return_updated_globals=True,
        residual_edge_update=True, residual_node_update=False, residual_global_update=False,
        update_edges_input=[True, True, True, False], # [edges, nodes_in, nodes_out, globals_]
        update_nodes_input=[True, False, False], # [aggregated_edges, nodes, globals_]
        update_global_input=[False, True, False], # [aggregated_edges, aggregated_nodes, globals_]
        **kwargs
    ):
        super().__init__()
        self.edge_mlp = MLP(**edge_mlp) if edge_mlp is not None else None
        self.node_mlp = MLP(**node_mlp) if node_mlp is not None else None

        self.aggregate_edges_local_ = aggregate_edges_local
        return

    def forward(self, h_edge, h_node, h_global, edge_indices, batch):
        h_node_in = h_node[edge_indices[0]]
        h_node_out = h_node[edge_indices[1]]

        features_to_concat = [h_edge, h_node_in, h_node_out]
        concat_features = torch.cat(features_to_concat, dim=-1)
        h_edge_new = self.edge_mlp(concat_features)

        edge_receiving_index = edge_indices[0]
        num_node = h_node.size()[0]
        aggregated_edges = scatter(h_edge_new, edge_receiving_index, dim=0, dim_size=num_node, reduce=self.aggregate_edges_local_)
        h_node_new = h_node + self.node_mlp(aggregated_edges)
        
        return h_edge_new, h_node_new, None, None
