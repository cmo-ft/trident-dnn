import torch
import torch_geometric

'''
    ParticleNet Implementation
'''

class ParticleStaticEdgeConv(torch_geometric.nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ParticleStaticEdgeConv, self).__init__(aggr='max')
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels + in_channels, out_channels[0], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[0], out_channels[1], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[1]),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels[1], out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
            torch.nn.ReLU()
        )
    def forward(self, x, edge_index, k, u=None):

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x, u=u)

    def message(self, edge_index, x_i, x_j, u_i, u_j):
        tmp = torch.cat([u_i, x_i, x_j - x_i], dim = 1)

        out_mlp = self.mlp(tmp)

        return out_mlp

    def update(self, aggr_out):
        return aggr_out

class ParticleDynamicEdgeConv(ParticleStaticEdgeConv):
    def __init__(self, in_channels, out_channels, k=7):
        super(ParticleDynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k
        self.skip_mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, out_channels[2], bias=False),
            torch_geometric.nn.BatchNorm(out_channels[2]),
        )
        self.act = torch.nn.ReLU()

    def forward(self, pts, fts, batch=None, u=None):
        edges = torch_geometric.nn.knn_graph(pts, self.k, batch, loop=False, flow=self.flow)
        aggrg = super(ParticleDynamicEdgeConv, self).forward(fts, edges, self.k, u[batch])
        # aggrg = self.gcns(fts, edges)
        x = self.skip_mlp(fts)
        out = torch.add(aggrg, x)
        return self.act(out)


class TridentNet(torch.nn.Module):
    def __init__(self, settings, DEVICE):
        super().__init__()
        previous_output_shape = settings['input_features']

        self.input_bn = torch_geometric.nn.BatchNorm(settings['input_features'])

        self.global_process = torch.nn.ModuleList()

        self.global_process.append(
                torch.nn.Sequential(
                    torch.nn.Linear(previous_output_shape, previous_output_shape),
                    torch_geometric.nn.BatchNorm(previous_output_shape),
                    torch.nn.ReLU()
                )
            )


        self.conv_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['conv_params']):
            K, channels = layer_param
            self.conv_process.append(ParticleDynamicEdgeConv(previous_output_shape, channels, k=K).to(DEVICE))
            
            self.global_process.append(
                torch.nn.Sequential(
                    torch.nn.Linear(previous_output_shape + channels[-1], channels[-1]),
                    torch_geometric.nn.BatchNorm(channels[-1]),
                    torch.nn.ReLU()
                )
            )
            
            previous_output_shape = channels[-1]


        self.fc_process = torch.nn.ModuleList()
        for layer_idx, layer_param in enumerate(settings['fc_params']):
            drop_rate, units = layer_param
            seq = torch.nn.Sequential(
                torch.nn.Linear(previous_output_shape, units),
                torch.nn.Dropout(p=drop_rate),
                torch.nn.ReLU()
            ).to(DEVICE)
            self.fc_process.append(seq)
            previous_output_shape = units



        self.output_mlp_linear = torch.nn.Linear(previous_output_shape, settings['output_classes'])
        self.output_activation = torch.nn.Softmax(dim=1)

    def forward(self, batch):
        fts = self.input_bn(batch.x)
        pts = batch.pos

        u = torch_geometric.nn.global_mean_pool(fts, batch.batch)
        u = self.global_process[0](u)

        for idx, layer in enumerate(self.conv_process):
            fts = layer(pts, fts, batch=batch.batch, u=u)
            u = torch.cat([u, torch_geometric.nn.global_mean_pool(fts, batch.batch)], dim = 1)
            u = self.global_process[idx+1](u)
            pts = fts

        # x = torch_geometric.nn.global_mean_pool(fts, batch.batch)
        x = u
        # x = fts

        for layer in self.fc_process:
            x = layer(x)

        x = self.output_mlp_linear(x)
        # x = self.output_activation(x)
        return x


import torch_cluster
import scipy.spatial
def knn(x, y, k, batch_x=None, batch_y=None):
    r"""Finds for each element in :obj:`y` the :obj:`k` nearest points in
    :obj:`x`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        y (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{M \times F}`.
        k (int): The number of neighbors.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^M`, which assigns each
            node to a specific example. (default: :obj:`None`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import knn

    .. testcode::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch_x = torch.tensor([0, 0, 0, 0])
        >>> y = torch.Tensor([[-1, 0], [1, 0]])
        >>> batch_x = torch.tensor([0, 0])
        >>> assign_index = knn(x, y, 2, batch_x, batch_y)
    """

    if batch_x is None:
        batch_x = x.new_zeros(x.size(0), dtype=torch.long)

    if batch_y is None:
        batch_y = y.new_zeros(y.size(0), dtype=torch.long)

    x = x.view(-1, 1) if x.dim() == 1 else x
    y = y.view(-1, 1) if y.dim() == 1 else y

    assert x.dim() == 2 and batch_x.dim() == 1
    assert y.dim() == 2 and batch_y.dim() == 1
    assert x.size(1) == y.size(1)
    assert x.size(0) == batch_x.size(0)
    assert y.size(0) == batch_y.size(0)

    if x.is_cuda:
        return torch_cluster.knn_cuda.knn(x, y, k, batch_x, batch_y)

    # Rescale x and y.
    min_xy = min(x.min().item(), y.min().item())
    x, y = x - min_xy, y - min_xy

    max_xy = max(x.max().item(), y.max().item())
    x, y, = x / max_xy, y / max_xy

    # Concat batch/features to ensure no cross-links between examples exist.
    x = torch.cat([x, 2 * x.size(1) * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
    y = torch.cat([y, 2 * y.size(1) * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

    tree = scipy.spatial.cKDTree(x.detach().numpy())
    dist, col = tree.query(
        y.detach().cpu(), k=k, distance_upper_bound=x.size(1))
    dist = torch.from_numpy(dist).to(x.dtype)
    col = torch.from_numpy(col).to(torch.long)
    row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
    mask = 1 - torch.isinf(dist).view(-1)
    row, col = row.view(-1)[mask], col.view(-1)[mask]

    return torch.stack([row, col], dim=0)

    
def knn_graph(x, k, batch=None, loop=False, flow='source_to_target'):
    r"""Computes graph edges to the nearest :obj:`k` points.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        k (int): The number of neighbors.
        batch (LongTensor, optional): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. (default: :obj:`None`)
        loop (bool, optional): If :obj:`True`, the graph will contain
            self-loops. (default: :obj:`False`)
        flow (string, optional): The flow direction when using in combination
            with message passing (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: :class:`LongTensor`

    .. testsetup::

        import torch
        from torch_cluster import knn_graph

    .. testcode::

        >>> x = torch.Tensor([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        >>> batch = torch.tensor([0, 0, 0, 0])
        >>> edge_index = knn_graph(x, k=2, batch=batch, loop=False)
    """

    assert flow in ['source_to_target', 'target_to_source']
    row, col = knn(x, x, k if loop else k + 1, batch, batch)
    row, col = (col, row) if flow == 'source_to_target' else (row, col)
    if not loop:
        mask = row != col
        row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)
