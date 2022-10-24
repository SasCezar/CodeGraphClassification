import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, BCEWithLogitsLoss
from torch_geometric.nn import GINConv, global_add_pool


class SimpleGIN(pl.LightningModule):
    def __init__(self, num_features, num_classes):
        super(SimpleGIN, self).__init__()

        num_features = num_features
        dim = 32

        nn1 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1 = Linear(dim, dim)
        self.fc2 = Linear(dim, num_classes)

        self.loss = BCEWithLogitsLoss()

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, data.edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, data.edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, data.edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, data.edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

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
