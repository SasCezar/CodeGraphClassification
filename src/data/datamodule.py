from typing import Optional

from loguru import logger
from pandas import DataFrame
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data.dataset import GitRankingDataset

'''
A DataModule standardizes the training, val, test splits, data preparation and transforms. 
The main advantage is consistent data splits, data preparation and transforms across models.
This allows you to share a full dataset without explaining how to download, split, 
transform, and process the data
'''


class GitRankingDataModule(pl.LightningDataModule):
    def __init__(self, graph_dir: str, feature_dir: str, out_dir: str, projects: DataFrame,
                 embedding_size: int, batch_size: int, num_splits: int, split: int):
        super().__init__()
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
        self.projects_test = None
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
        if stage == 'test' or stage is None:
            self.test_data = GitRankingDataset(graph_dir=self.graph_dir, feature_dir=self.feature_dir,
                                               projects=self.projects_test, out_dir=self.root,
                                               embedding_size=self.embedding_size, transform=self.transform,
                                               pre_transform=self.pre_transform, pre_filter=self.pre_filter)
        else:
            raise ValueError(f"Stage {stage} not recognized.")

    def train_dataloader(self):
        return DataLoader(self.train_data, shuffle=True, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_data, shuffle=False, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, shuffle=False, batch_size=self.batch_size)

    def create_splits(self):
        kf = GroupKFold(n_splits=self.num_splits)
        gss = GroupShuffleSplit(n_splits=1, train_size=.8)
        all_splits = list(kf.split(self.projects, groups=self.projects['name'].tolist()))
        temp_indexes, val_indexes = all_splits[self.split]
        temp_indexes, val_indexes = temp_indexes.tolist(), val_indexes.tolist()
        projects_temp, self.projects_val = self.projects.iloc[temp_indexes], self.projects.iloc[val_indexes]
        indexes = list(gss.split(projects_temp, groups=projects_temp['name'].tolist()))
        logger.info(f"Indexes: {indexes}")
        train_indexes, test_indexes = indexes[0]
        self.projects_train, self.projects_test = projects_temp.iloc[train_indexes], projects_temp.iloc[test_indexes]
        logger.info(f"Train: {len(self.projects_train)} projects, Val: {len(self.projects_val)} projects, "
                    f"Test: {len(self.projects_test)} projects")


    @property
    def num_node_features(self) -> int:
        return self.embedding_size

    @property
    def num_classes(self) -> int:
        return len(set())
