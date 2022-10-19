import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool, PNAConv, BatchNorm


class PNA(pl.LightningModule):
    def __init__(self, node, out, deg):
        super(PNA, self).__init__()

        self.node_emb = Embedding(node, 75)
        # self.edge_emb = Embedding(4, 50)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=75, out_channels=75,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(75))

        self.mlp = Sequential(Linear(75, 50), ReLU(), Linear(50, 25), ReLU(),
                              Linear(25, out))

    def forward(self, x, edge_index, batch):
        x = self.node_emb(x.squeeze())

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, edge_index)))

        x = global_add_pool(x, batch)
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        x, edge_index, batch = batch
        out = self.forward(x, edge_index, batch)
        loss = F.nll_loss(out, batch.y)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, edge_index, batch = batch
        out = self.forward(x, edge_index, batch)
        loss = F.nll_loss(out, batch.y)
        self.log('val/loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, edge_index, batch = batch
        out = self.forward(x, edge_index, batch)
        loss = F.nll_loss(out, batch.y)
        self.log('test/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)