import torch
import torch.nn as nn
import pytorch_lightning as pl
from xformers.factory.model_factory import xFormer, xFormerConfig


class xCaptionModel(pl.LightningModule):
    def __init__(
            self,
            seq_len,
            vocab_size,
            proj_dim,
            intermidiate_dim,
            embed_dim,
            nhead,
            hidden_layer_multiplier,
            activation,
            num_layers,
            dropout,
            learning_rate,
            betas,
            weight_decay,
            label_smoothing,
            batch_size,
            dataset_steps,
            max_epochs
    ):
        super().__init__()
        self.save_hyperparameters()

        config = [
            {
                "block_type": "decoder",
                "reversible": False,
                "num_layers": self.hparams.num_layers, 
                "dim_model": self.hparams.embed_dim,
                "residual_norm_style": "pre",
                "position_encoding_config": {
                    "name": "vocab",
                    "seq_len": self.hparams.seq_len,
                    "vocab_size": self.hparams.vocab_size,
                },
                "multi_head_config_masked": {
                    "num_heads": self.hparams.nhead,
                    "residual_dropout": self.hparams.dropout,
                    "use_rotary_embeddings": True,
                    "attention": {
                        "name": "scaled_dot_product",
                        "dropout": self.hparams.dropout,
                        "causal": True,
                        "seq_len": self.hparams.seq_len,
                    },
                },
                "multi_head_config_cross": {
                    "num_heads": self.hparams.nhead,
                    "residual_dropout": self.hparams.dropout,
                    "attention": {
                        "name": "scaled_dot_product",
                        "dropout": self.hparams.dropout,
                        "causal": True,
                        "seq_len": self.hparams.seq_len,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dropout": self.hparams.dropout,
                    "activation": self.hparams.activation,
                    "hidden_layer_multiplier": self.hparams.hidden_layer_multiplier,
                },
            },
        ]
        # extracted features -> intermediate memory
        self.linear_1 = nn.Linear(self.hparams.proj_dim, self.hparams.intermidiate_dim)
        self.layer_norm_1 = nn.LayerNorm(self.hparams.intermidiate_dim)
        self.gelu_1 = nn.GELU()
        self.dropout_1 = nn.Dropout(self.hparams.dropout)

        # intermediate memory -> embeddings memory
        self.linear_2 = nn.Linear(self.hparams.intermidiate_dim, self.hparams.embed_dim)
        self.layer_norm_2 = nn.LayerNorm(self.hparams.embed_dim)
        self.gelu_2 = nn.GELU()
        self.dropout_2 = nn.Dropout(self.hparams.dropout)

        # embeddings memory -> captions
        xformer_config = xFormerConfig(config)
        self.xformer = xFormer.from_config(xformer_config)
        self.layer_norm_3 = nn.LayerNorm(self.hparams.embed_dim)
        self.linear = nn.Linear(self.hparams.embed_dim, self.hparams.vocab_size)
        self.dropout_3 = nn.Dropout(self.hparams.dropout)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)

    def forward(self, memory, tgt):
        memory = self.linear_1(memory)
        memory = self.layer_norm_1(memory)
        memory = self.gelu_1(memory)
        memory = self.dropout_1(memory)

        memory = self.linear_2(memory)
        memory = self.layer_norm_2(memory)
        memory = self.gelu_2(memory)
        memory = self.dropout_2(memory)
        memory = memory.unsqueeze(1).repeat(1, self.hparams.seq_len, 1)

        tgt = self.xformer(memory, tgt)
        tgt = self.layer_norm_3(tgt)
        tgt = self.linear(tgt)
        tgt = self.dropout_3(tgt)
        return tgt
    
    
    def _common_step(self, batch):
        _, images, captions = batch
        src, tgt = captions[:, :-1], captions[:, 1:]
        logits = self(images, src)
        loss = self.loss_fn(logits.permute(0, 2, 1), tgt)
        return loss
    

    def training_step(self, batch, _):
        loss = self._common_step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, _):
        loss = self._common_step(batch)
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        decay = []
        no_decay = []
        no_decay_layers = ["bias", "norm", "embeddings"]

        # Separate parameters into decay and no_decay groups.
        for name, param in self.named_parameters():
            if any(nd in name for nd in no_decay_layers):
                no_decay.append(param)
            else:
                decay.append(param)

        # Define optimizer groups with weight decay settings.
        optim_groups = [
            {"params": decay, "weight_decay": self.hparams.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]

        # Create AdamW optimizer with specified settings.
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
        )

        # Define learning rate scheduler settings.
        lr_scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.hparams.dataset_steps * self.hparams.max_epochs // self.hparams.batch_size,
                eta_min=0
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }

        # Return the optimizer and learning rate scheduler.
        return [optimizer], [lr_scheduler]
