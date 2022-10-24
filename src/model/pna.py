import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, BCEWithLogitsLoss
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_add_pool, PNAConv, BatchNorm


class PNA(pl.LightningModule):
    def __init__(self, in_channels, hidden, num_classes, deg):
        super(PNA, self).__init__()

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(4):
            conv = PNAConv(in_channels=in_channels, out_channels=hidden,
                           aggregators=aggregators, scalers=scalers, deg=deg,
                           towers=5, pre_layers=1, post_layers=1,
                           divide_input=False)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(hidden))

        self.mlp = Sequential(Linear(hidden, hidden), ReLU(), Linear(hidden, hidden), ReLU(),
                              Linear(hidden, num_classes))

        self.loss = BCEWithLogitsLoss()

    def forward(self, data):
        x = data.x

        for conv, batch_norm in zip(self.convs, self.batch_norms):
            x = F.relu(batch_norm(conv(x, data.edge_index)))

        x = global_add_pool(x, data.batch)
        return self.mlp(x)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self.loss(out, batch.y)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self.loss(out, batch.y)
        self.log('val/loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self.loss(out, batch.y)
        self.log('test/loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
