from typing import Optional

from pandas import DataFrame
from sklearn.model_selection import GroupKFold
from torch_geometric.data import DataLoader
from torch_geometric.data.lightning_datamodule import LightningDataModule

from data.dataset import GitRankingDataset

'''
A DataModule standardizes the training, val, test splits, data preparation and transforms. 
The main advantage is consistent data splits, data preparation and transforms across models.
This allows you to share a full dataset without explaining how to download, split, 
transform, and process the data
'''


class GitRankingDataModule(LightningDataModule):
    def __init__(self, graph_dir: str, feature_dir: str, out_dir: str,
                 projects: DataFrame, embedding_size: int, batch_size: int,
                 num_splits: int, split: int, has_val: bool = True, has_test: bool = True, **kwargs):
        super().__init__(has_val, has_test, **kwargs)
        self.root = out_dir
        self.graph_dir = graph_dir
        self.feature_dir = feature_dir
        self.projects = projects
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        self.num_splits = num_splits
        self.split = split

        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.projects_train = None
        self.projects_val = None
        self.create_splits()

        self.transform = None
        self.pre_transform = None
        self.pre_filter = None

    def prepare_data(self):
        GitRankingDataset(graph_dir=self.graph_dir, feature_dir=self.feature_dir,
                          projects=self.projects, out_dir=self.root,
                          embedding_size=self.embedding_size, transform=self.transform,
                          pre_transform=self.pre_transform, pre_filter=self.pre_filter)

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_data = GitRankingDataset(graph_dir=self.graph_dir, feature_dir=self.feature_dir,
                                                projects=self.projects_train, out_dir=self.root,
                                                embedding_size=self.embedding_size, transform=self.transform,
                                                pre_transform=self.pre_transform, pre_filter=self.pre_filter)
            self.val_data = GitRankingDataset(graph_dir=self.graph_dir, feature_dir=self.feature_dir,
                                              projects=self.projects_val, out_dir=self.root,
                                              embedding_size=self.embedding_size, transform=self.transform,
                                              pre_transform=self.pre_transform, pre_filter=self.pre_filter)
        # if stage == 'test' or stage is None:
        #     self.test_data = ArcanDependenciesDataset(graph_dir=self.graph_dir, feature_dir=self.feature_dir,
        #                                               projects=self.projects_test, out_dir=self.root,
        #                                               embedding_size=self.embedding_size, transform=self.transform,
        #                                               pre_transform=self.pre_transform, pre_filter=self.pre_filter)
        else:
            raise ValueError(f"Stage {stage} not recognized.")

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=False, batch_size=self.batch_size)

    # def test_dataloader(self):
    #     return DataLoader(self.test_data, shuffle=False, batch_size=self.batch_size)

    def create_splits(self):
        kf = GroupKFold(n_splits=self.num_splits)
        all_splits = [k for k in kf.split(self.projects, groups=self.projects['name'].tolist())]
        train_indexes, val_indexes = all_splits[self.split]
        train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
        self.projects_train, self.projects_val = self.projects.iloc[train_indexes], self.projects.iloc[val_indexes]

    @property
    def num_node_features(self) -> int:
        return self.embedding_size

    @property
    def num_classes(self) -> int:
        return len(set())
