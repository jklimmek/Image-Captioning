import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from tokenizers import Tokenizer
from .model import xCaptionModel


class ImageCaptioner:
    def __init__(self, captioner, tokenizer, vit, vit_transforms, maxlen, device):
        """
        Initialize an ImageCaptioner instance.

        Args:
            captioner (torch.nn.Module): The captioning model.
            tokenizer (Tokenizer): The tokenizer for encoding and decoding captions.
            vit (torch.nn.Module): The vision transformer model.
            vit_transforms (torchvision.transforms.Compose): Image transformations for the vision transformer.
            maxlen (int): The maximum length of generated captions.
            device (str or torch.device): The device to perform computations on.
        """
        self.captioner = captioner
        self.tokenizer = tokenizer
        self.vit = vit
        self.vit_transforms = vit_transforms
        self.maxlen = maxlen
        self.device = device


    @classmethod
    def from_files(cls, checkpoint_path, tokenizer_path, maxlen=40, device="cpu"):
        """
        Load an ImageCaptioner instance from saved model checkpoint and tokenizer files.

        Args:
            checkpoint_path (str): The path to the model checkpoint file.
            tokenizer_path (str): The path to the tokenizer file.
            maxlen (int, optional): The maximum length of generated captions. Defaults to 40.
            device (str or torch.device, optional): The device to perform computations on. Defaults to "cpu".

        Returns:
            ImageCaptioner: The loaded ImageCaptioner instance.
        """
        model = cls._load_captioner(checkpoint_path, device)
        tokenizer = cls._load_tokenizer(tokenizer_path, maxlen)
        vit, vit_transforms = cls._load_extractor_components(device)
        return cls(model, tokenizer, vit, vit_transforms, maxlen, device)
    

    @torch.no_grad()
    def generate_caption(self, image, beam_size=10):
        """
        Generate a caption for the given image using beam search.

        Args:
            image (PIL.Image.Image or torch.Tensor): The input image.
            beam_size (int, optional): The size of the beam for beam search. Default is 10.

        Returns:
            str: The generated caption.
        """
        # Preprocess the image.
        image = self.vit_transforms(image).unsqueeze(0).to(self.device)

        # Extract image features using the Vision Transformer model.
        image_features = self.vit(image)

        # Initialize the caption with a start token.
        caption = torch.zeros((1, self.maxlen), dtype=torch.long, device=self.device)
        caption[:, 0] = self.tokenizer.token_to_id("[SOS]")

        # Initialize beam candidates.
        beam_candidates = [{'caption': caption, 'score': 0.0}]

        # Generate the caption using beam search.
        for i in range(1, self.maxlen):
            next_beam_candidates = []

            # Expand each beam candidate.
            for candidate in beam_candidates:
                partial_caption = candidate['caption']

                # Generate logits for the next token using the captioner model.
                logits = self.captioner(memory=image_features, tgt=partial_caption)

                # Compute probabilities using softmax.
                probabilities = F.softmax(logits[0, i-1], dim=-1)

                # Select top-k tokens based on probabilities.
                if beam_size > 1:
                    top_tokens = probabilities.topk(beam_size)[1].squeeze()
                else:
                    top_token = probabilities.argmax()
                    top_tokens = torch.tensor([top_token.item()], device=self.device)

                # Create new candidates by appending each token to the partial caption.
                for token in top_tokens:
                    new_caption = partial_caption.clone()
                    new_caption[:, i] = token.item()
                    new_score = candidate['score'] + torch.log(probabilities[token])
                    next_beam_candidates.append({'caption': new_caption, 'score': new_score})

            # Sort the new beam candidates based on score and select top-k candidates.
            next_beam_candidates.sort(key=lambda x: x['score'], reverse=True)
            beam_candidates = next_beam_candidates[:beam_size]

            # Check if all of the candidates have ended with the end token.
            if all(candidate['caption'][:, i].item() == self.tokenizer.token_to_id("[EOS]") for candidate in beam_candidates):
                break

        # Retrieve the best caption from the top candidate.
        best_caption = beam_candidates[0]['caption']

        # Decode the caption into a string.
        generated_caption = self.tokenizer.decode(best_caption[0].cpu().numpy())

        # Return the generated caption.
        return generated_caption


    def _load_captioner(checkpoint_path, device):
        """
        Load the pre-trained captioning model from the given checkpoint file.

        Args:
            checkpoint_path (str): The path to the checkpoint file.
            device (str or torch.device): The device (CPU or GPU) to load the model on.

        Returns:
            xCaptionModel: The loaded pre-trained captioning model.
        """
        model = xCaptionModel.load_from_checkpoint(checkpoint_path)
        model.to(device)
        model.eval()
        return model


    def _load_tokenizer(tokenizer_name, max_length):
        """
        Load the tokenizer from the given file and set up padding and truncation.

        Args:
            tokenizer_name (str): The file containing the tokenizer.
            max_length (int): The maximum length of input sequences for padding and truncation.

        Returns:
            Tokenizer: The loaded tokenizer with padding and truncation enabled.
        """
        tokenizer = Tokenizer.from_file(tokenizer_name)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[PAD]"), length=max_length)
        tokenizer.enable_truncation(max_length)
        return tokenizer


    def _load_extractor_components(device):
        """
        Load the Vision Transformer (ViT) model and transformation components for image extraction.

        Args:
            device (torch.device): The device (CPU or GPU) to load the model on.

        Returns:
            tuple: A tuple containing the loaded ViT model and the image transformation pipeline.
                (vit_model, vit_transforms)
        """
        vit = models.vision_transformer.vit_b_16(weights="IMAGENET1K_V1")
        vit.to(device)
        vit.eval()
        vit.heads = nn.Identity()
        vit_transforms = transforms.Compose(
            [
                # Transforms specific to the chosen ViT model.
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        ) 
        return vit, vit_transforms