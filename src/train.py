import hydra
import pandas
import pytorch_lightning as pl
from hydra.utils import instantiate
from more_itertools import flatten
from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.callbacks.logging import TBParametersLoggingCallback
from utils import encode_labels


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def train(cfg: DictConfig):
    seed_everything(cfg.seed)
    projects = pandas.read_csv(cfg.project_path)
    projects = encode_labels(projects)
    num_classes = len(set(flatten(projects["label"].tolist())))
    data_module = instantiate(cfg.datamodule, projects=projects)
    model = instantiate(cfg.model, num_classes=num_classes, num_features=300)

    train_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{train_loss:.1f}",
        monitor='train/loss',
        mode='min',
        save_top_k=3
    )

    latest_checkpoint = ModelCheckpoint(
        filename='latest-{epoch}-{step}',
        every_n_train_steps=500,
        save_top_k=1
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode="min"
    )

    tb_logger = pl_loggers.TensorBoardLogger("logs/", log_graph=True)
    tb_param_logger = TBParametersLoggingCallback()
    trainer = pl.Trainer(max_epochs=1,
                         callbacks=[latest_checkpoint, train_checkpoint, tb_param_logger, early_stop_callback],
                         gpus=1,
                         logger=tb_logger,
                         fast_dev_run=False,
                         val_check_interval=0.3)

    trainer.fit(model, data_module)
    trainer.validate(model, data_module)


if __name__ == '__main__':
    train()
