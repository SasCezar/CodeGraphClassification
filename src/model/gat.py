import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch.nn import Linear, BCEWithLogitsLoss
from torch_geometric.nn import global_mean_pool, GATConv


class GATNetwork(pl.LightningModule):
    """
    Three layer Graph Convolutional Network (GCN) with Morris et al. convolution (https://arxiv.org/abs/1810.02244)
    """

    def __init__(self, num_features, hidden_channels, num_classes):
        super(GATNetwork, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=8, concat=True)
        self.conv2 = GATConv(hidden_channels, hidden_channels,  heads=8, concat=True)
        self.conv3 = GATConv(hidden_channels, hidden_channels,  heads=8, concat=True)
        self.lin = Linear(hidden_channels * 8, num_classes)

        self.accuracy = torchmetrics.Accuracy()

        self.loss = BCEWithLogitsLoss()

    def forward(self, data):
        # 1. Obtain node embeddings
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = self.conv2(x, data.edge_index)
        x = x.relu()
        x = self.conv3(x, data.edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, data.batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return F.log_softmax(x, dim=-1)

    def training_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self.loss(out, batch.y)
        self.log('train/loss', loss, batch_size=64)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self.loss(out, batch.y)
        self.log('val/loss', loss, batch_size=64)
        return loss

    def test_step(self, batch, batch_idx):
        out = self.forward(batch)
        loss = self.loss(out, batch.y)
        self.log('test/loss', loss, batch_size=64)
        return loss

    def configure_optimizers(self):

        return torch.optim.RMSprop(self.parameters(), lr=0.001)
