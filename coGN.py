
import torch
from torch import nn
from torch_scatter import scatter

from .edge_embedding import EdgeEmbedding
from .layers import coGNLayer, MLP


class coGN(nn.Module):
    def __init__(
        self,
        node_class=None,
        emb_dim=128,
        extra_atom_features=True,
        num_layer=5,
        bins_distance=32,
        distance_cutoff=5,
    ):
        super(coGN, self).__init__()
        self.emb_dim = emb_dim

        ########## input layer ##########
        self.extra_atom_features = extra_atom_features
        if extra_atom_features:
            # This is only required when we use extra atom features (in crystals)
            from Geom3D.models import ExtraAtomEmbedding
            self.atom_embedding = ExtraAtomEmbedding(node_class, emb_dim)
        else:
            self.atom_embedding = nn.Embedding(node_class, emb_dim)
        self.atom_mlp = nn.Linear(emb_dim, emb_dim)

        self.edge_embedding = EdgeEmbedding(
            bins_distance=bins_distance,
            max_distance=distance_cutoff,
            distance_log_base=1.,
            bins_voronoi_area=None,
            max_voronoi_area=None)
        self.edge_mlp = nn.Linear(bins_distance, emb_dim)

        processing_block_cfg = {
            'edge_mlp': {'input_dim': emb_dim*3, 'hidden_dim_list': [emb_dim] * 5, 'activation': 'silu'},
            'node_mlp': {'input_dim': emb_dim*1, 'hidden_dim_list': [emb_dim] * 1, 'activation': 'silu'},
            'global_mlp': None,
            'aggregate_edges_local': 'sum',
        }

        output_block_cfg = {
            'edge_mlp': None,
            'node_mlp': None,
            'global_mlp': {'input_dim': emb_dim * 1, 'hidden_dim_list': [1], 'activation': None,},
        }

        ########## processing layer ##########
        self.num_layer = num_layer
        self.processing_layers = nn.ModuleList()
        for _ in range(self.num_layer):
            self.processing_layers.append(coGNLayer(**processing_block_cfg))

        ########## output layer ##########
        self.output_layer = MLP(**output_block_cfg['global_mlp'])

        return
    
    def forward(self, z, pos, batch, edge_index, return_latent=False):
        h_node = self.atom_embedding(z)
        h_node = self.atom_mlp(h_node)

        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        h_edge = self.edge_embedding(edge_weight)
        h_edge = self.edge_mlp(h_edge)

        for layer_idx in range(self.num_layer):
            h_edge, h_node, _, _ = self.processing_layers[layer_idx](h_edge, h_node, None, edge_index, batch)
            
        num_graph = batch.max().item() + 1
        h_graph = scatter(h_node, batch, dim=0, dim_size=num_graph, reduce="mean")
        out = self.output_layer(h_graph)
        return out

    def forward_with_gathered_index(self, gathered_z, pos, batch, edge_index, gathered_batch, periodic_index_mapping, return_latent=False,):
        h_node = self.atom_embedding(gathered_z)
        h_node = self.atom_mlp(h_node)

        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
        h_edge = self.edge_embedding(edge_weight)
        h_edge = self.edge_mlp(h_edge)

        gathered_row = periodic_index_mapping[row]
        gathered_col = periodic_index_mapping[col]
        gathered_edge_index = torch.stack([gathered_row, gathered_col])

        for layer_idx in range(self.num_layer):
            h_edge, h_node, _, _ = self.processing_layers[layer_idx](h_edge, h_node, None, gathered_edge_index, gathered_batch)

        num_graph = gathered_batch.max().item() + 1
        h_graph = scatter(h_node, gathered_batch, dim=0, dim_size=num_graph, reduce="mean")
        out = self.output_layer(h_graph)
        return out
