import pytorch_lightning as pl

from config.config import ex
from data.DataModule import SourceRetrievalDataModule
from model.Model import SourceRetrievalModule
import copy
import os
from accelerate import Accelerator
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.strategies.fsdp import FSDPStrategy

import warnings
warnings.filterwarnings(action='ignore')

@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    
    
    # Print config
    for key, val in _config.items():
        key_str = "{}".format(key) + (" " * (30 - len(key)))
        print(f"{key_str}   =   {val}")    
    
    pl.seed_everything(_config["seed"])   
    
    exp_name = f'{_config["exp_name"]}'

    os.makedirs(_config["log_dir"], exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        # save_top_k=3,
        verbose=True,
        monitor="val/mean_ROUGE",
        filename='epoch={epoch}-step={step}-val_mean_ROUGE={val/mean_ROUGE:.5f}',
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
    )
    logger = pl.loggers.TensorBoardLogger(
        _config["log_dir"],
        name=f'{exp_name}_seed{_config["seed"]}_from_{_config["load_path"].split("/")[-1][:-5]}',
    )

    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")
    
    callbacks = [checkpoint_callback, lr_callback]

    accumulate_grad_batches = max(_config["batch_size"] // (
        _config["per_gpu_batch_size"] * len(_config['gpus']) * _config["num_nodes"]
    ), 1)

    dm = SourceRetrievalDataModule(_config=_config)
    print("loaded data")
    if _config['mode'] == 'test':
        model = SourceRetrievalModule.load_from_checkpoint(_config["load_path"], _config=_config)
    else:
        model = SourceRetrievalModule(_config=_config)
        # model = SourceRetrievalModule.load_from_checkpoint(_config["load_path"], _config=_config, map_location="cpu")
    print("loaded model")
    trainer = pl.Trainer(
        devices=_config['gpus'],
        max_steps=_config["max_steps"],
        accelerator="gpu",
        strategy= DDPStrategy(), # DeepSpeedStrategy(logging_batch_size_per_gpu=_config["per_gpu_batch_size"]),
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        accumulate_grad_batches=accumulate_grad_batches,
        val_check_interval=_config['val_check_interval'],
        gradient_clip_val=1.0,
        reload_dataloaders_every_n_epochs=1,
        )

    if _config['mode'] == 'test':
        trainer.test(model, datamodule=dm)
    else:
        trainer.fit(model, datamodule=dm)