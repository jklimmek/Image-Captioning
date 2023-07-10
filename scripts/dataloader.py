import pytorch_lightning as pl
from torch.utils.data import DataLoader


class CaptionDataTrainModule(pl.LightningDataModule):
    def __init__(
            self, 
            train_dataset, 
            val_dataset, 
            batch_size, 
            num_workers
        ):
        """
        LightningDataModule for caption training data.

        Args:
            train_dataset (Dataset): The training dataset.
            val_dataset (Dataset): The validation dataset.
            batch_size (int): The batch size for the data loaders.
            num_workers (int): The number of workers for data loading.
        """
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        """
        Get the data loader for training.

        Returns:
            DataLoader: The data loader for training data.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
        )
    
    def val_dataloader(self):
        """
        Get the data loader for validation.

        Returns:
            DataLoader: The data loader for validation data.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
        )
