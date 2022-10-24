import hydra
import pandas
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning import loggers as pl_loggers, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from data.datamodule import GitRankingDataModule
from src.callbacks.logging import TBParametersLoggingCallback
from utils import encode_labels, only_top_labels


@hydra.main(config_path="conf", config_name="main", version_base="1.2")
def train(cfg: DictConfig):
    seed_everything(cfg.seed)
    projects = pandas.read_csv(cfg.project_path)
    projects = only_top_labels(projects, 10)
    projects = encode_labels(projects)
    data_module: GitRankingDataModule = instantiate(cfg.datamodule, projects=projects)
    print(data_module.num_classes)
    model = instantiate(cfg.model, num_classes=data_module.num_classes, num_features=300)

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

    tb_logger = pl_loggers.TensorBoardLogger("../logs/", log_graph=True)
    tb_param_logger = TBParametersLoggingCallback()
    trainer = pl.Trainer(callbacks=[latest_checkpoint, train_checkpoint, tb_param_logger],
                         accelerator='gpu',
                         devices=1,
                         logger=tb_logger,
                         fast_dev_run=False,
                         max_epochs=-1)

    data_module.setup("fit")
    data_module.setup("validate")
    trainer.fit(model, train_dataloaders=data_module.train_dataloader(),  val_dataloaders=data_module.val_dataloader())
    # trainer.validate(model, data_module)


if __name__ == '__main__':
    train()
