import os
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
from tokenizers import Tokenizer

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class PreprocessorDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Dataset class for preprocessing images and associated comments.

        Args:
            root_dir (str): The root directory where the images are located.
            csv_file (str): The path to the CSV file containing image names and comments.
            transform (callable, optional): Optional transform to be applied to the images. Default is None.
        """
        super().__init__()
        self.root_dir = root_dir
        self.csv_file = pd.read_csv(csv_file, sep="|")
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.csv_file.iloc[idx]["image_name"])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img_path, img, self.csv_file.iloc[idx]["comment"]    


class DecoderDataset(Dataset):
    def __init__(self, data):
        """
        Dataset class for decoding preprocessed data.

        Args:
            data (dict): A dictionary containing image names, tokenized comments, and features.
        """
        super().__init__()
        self.image_names = data["image_name"]
        self.tokenized_comments = data["tokenized_comment"]
        self.features = data["features"]

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        return self.image_names[idx], self.features[idx], self.tokenized_comments[idx]
    

def load_tokenizer(tokenizer_name, max_length):
    """
    Loads a tokenizer from a file and configures it for padding and truncation.

    Args:
        tokenizer_name (str): The name or path of the tokenizer file.
        max_length (int): The maximum sequence length.

    Returns:
        Tokenizer: The loaded tokenizer object.
    """
    tokenizer = Tokenizer.from_file(tokenizer_name)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), length=max_length + 1)
    tokenizer.enable_truncation(max_length + 1)
    return tokenizer


def load_extractor_components(device, augment_data=False):
    """
    Loads vision transformer components for feature extraction.

    Args:
        device (torch.device): The device to load the components on.
        augment_data (bool, optional): Whether to apply data augmentation. Defaults to False.

    Returns:
        transforms.Compose, torch.nn.Module: The composed image transformations and the vision transformer model.
    """
    vit = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
    vit = vit.to(device)
    vit.heads = nn.Identity()
    p = 0.0 if not augment_data else 1.0
    degrees = 0 if not augment_data else 10
    vit_transforms = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        # <---- Data Augmentation ---->
        transforms.RandomHorizontalFlip(p=p),
        transforms.RandomRotation(degrees=degrees, interpolation=transforms.InterpolationMode.BILINEAR),
        # <---- Data Augmentation ---->
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 
    return vit_transforms, vit
    

def configure_data_loader(dataset, batch_size):
    """
    Configure a data loader for the given dataset.

    Args:
        dataset (Dataset): The dataset to be loaded.
        batch_size (int): The batch size for the data loader.

    Returns:
        DataLoader: The configured data loader.
    """
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    return data_loader


@torch.no_grad()
def extract_features(data_loader, tokenizer, vit, device):
    """
    Extract image features and tokenize comments using a pre-trained model.

    Args:
        data_loader (DataLoader): The data loader for the dataset.
        tokenizer (Tokenizer): The tokenizer for tokenizing comments.
        vit (nn.Module): The pre-trained Vision Transformer model.
        device (str or torch.device): The device to run the extraction on.

    Returns:
        dict: A dictionary containing the image names, tokenized comments, and features.
    """
    # Initialize data.
    data = {
        "image_name": [], 
        "tokenized_comment": [], 
        "features": []
    }

    # Iterate over data.
    for batch in tqdm(data_loader, desc="Extracting features", total=len(data_loader), ncols=100):
        img_name, img, comment = batch

        # Extract features.
        img = img.to(device)
        features = vit(img).cpu()

        # Tokenizer comment.
        tokenized_comment = [t.ids for t in tokenizer.encode_batch(comment)]
        tokenized_comment = torch.tensor(tokenized_comment, dtype=torch.long)

        # Append data.
        img_names = [i.split("\\")[-1] for i in img_name]
        data["image_name"] += img_names
        data["tokenized_comment"] += tokenized_comment
        data["features"] += features
    return data


if __name__ == "__main__":
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Preprocess .csv file.")
    parser.add_argument("--csv-file", type=str, required=True, help="File to preprocess.")
    parser.add_argument("--img-dir", type=str, required=True, help="Image directory.")
    parser.add_argument("--tokenizer", type=str, required=True, help="Tokenizer file.")
    parser.add_argument("--seq-len", type=int, default=40, help="Maximum sequence length.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--output-file", type=str, required=True, help="Output directory.")
    parser.add_argument("--cuda", action="store_true", help="CUDA device to use.")
    args = parser.parse_args()

    # set device.
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # Load extractor components.
    vit_transforms, vit = load_extractor_components(device=device)

    # Create dataset.
    dataset = PreprocessorDataset(
        root_dir=args.img_dir, 
        csv_file=args.csv_file, 
        transform=vit_transforms
    )

    # Load tokenizer.
    tokenizer = load_tokenizer(
        tokenizer_name=args.tokenizer, 
        max_length=args.seq_len
    )
    
    # Configure data loader.
    data_loader = configure_data_loader(
        dataset=dataset,
        batch_size=args.batch_size
    )

    # Extract features.
    data = extract_features(
        data_loader=data_loader, 
        tokenizer=tokenizer, 
        vit=vit, 
        device=device
    )
    
    # Create decoder dataset.
    decoder_dataset = DecoderDataset(data)

    # Save decoder dataset.
    torch.save(decoder_dataset, args.output_file)
