import pytorch_lightning as pl


class TBParametersLoggingCallback(pl.Callback):
    """
    Logs all parameters to tensorboard
    """
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # iterating through all parameters
        for name, params in pl_module.named_parameters():
            pl_module.logger.experiment.add_histogram(name, params, trainer.current_epoch)