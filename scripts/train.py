import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from .dataloader import CaptionDataTrainModule
from .model import xCaptionModel
from .extractor import DecoderDataset
from .utils import yaml_to_dict, get_name

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")

# import tensorboard 


def main(hyperparams, config, comment=""):
    # Load dataset.
    train_dataset = torch.load(config["train_file"])
    dev_dataset = torch.load(config["dev_file"])
    data_loader = CaptionDataTrainModule(train_dataset, dev_dataset, hyperparams["batch_size"], num_workers=6)

    # Initialize model.
    hyperparams["dataset_steps"] = len(train_dataset)
    model = xCaptionModel(**hyperparams)

    # get name of model.
    model_name = get_name(model, **hyperparams)

    # Create callbacks.
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["runs_dir"], model_name), 
        filename="{epoch:02d}-{train_loss:.4f}-{val_loss:.4f}",
        save_top_k=-1
    )

    # Create logger.
    logger = TensorBoardLogger(
        save_dir=config["logs_dir"], 
        name=model_name + "_" + comment,
        default_hp_metric=False,
        version=""
    )
    
    # Create trainer.
    trainer = pl.Trainer(
        max_epochs=hyperparams["max_epochs"], 
        logger=logger,
        log_every_n_steps=5,
        callbacks=[lr_monitor, checkpoint_callback],
    )

    # Train model.
    ckpt_path = os.path.join(model_name, config["checkpoint_file"]) if config["checkpoint_file"] else None 
    trainer.fit(model, data_loader, ckpt_path=ckpt_path)


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("--hyperparams", type=str, required=True, help="Hyperparams YAML file.")
    parser.add_argument("--config", type=str, required=True, help="Config YAML file.")
    parser.add_argument("--comment", type=str, default="", help="Additional comment for Tensorbord.")
    args = parser.parse_args()

    # Read YAML files.
    hyperparams = yaml_to_dict(args.hyperparams)
    config = yaml_to_dict(args.config)

    # Run training.
    main(hyperparams, config, args.comment)