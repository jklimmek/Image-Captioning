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
    def generate_caption(
        self,
        image,
        beam_size=0,
        top_k=0,
        top_p=0.0,
        temperature=1.0,
        alpha=0.0
    ):
        """
        Generate a caption for the given image using different captioning strategies.
        Available strategies are: beam search, top-k sampling, top-p sampling and contrastive search.

        Args:
            image (PIL.Image.Image or torch.Tensor): The input image.
            beam_size (int, optional): The size of the beam for beam search. Defaults to 0 (disabled).
            top_k (int, optional): The number of top-k tokens to sample from. Defaults to 0 (disabled).
            top_p (float, optional): The cumulative probability threshold for top-p sampling. Defaults to 0 (disabled).
            temperature (float, optional): The temperature value for controlling the randomness of token sampling.
                                           Higher values generate more diversed output. Defaults to 1.0.
            alpha (float, optional): The degeneration penalty for contrastive search. Defaults to 0 (disabled).

        Returns:
            str: The generated caption.
        """
        # Preprocess the image and move it to the specified device.
        image = self.vit_transforms(image).unsqueeze(0).to(self.device)

        # Extract image features using the Vision Transformer model.
        image_features = self.vit(image)

        # Initialize the caption with a start token.
        empty_caption = torch.zeros((1, self.maxlen), dtype=torch.long, device=self.device)
        empty_caption[:, 0] = self.tokenizer.token_to_id("[SOS]")

        # Select captioning strategy based on provided arguments.
        if beam_size > 0 and top_k == 0 and top_p == 0:
            tokens = self._beam_search(image_features, empty_caption, beam_size=beam_size)
        elif top_k > 0 and top_p == 0 and alpha == 0:
            tokens = self._top_k(image_features, empty_caption, top_k=top_k, temperature=temperature)
        elif top_k == 0 and top_p > 0 and alpha == 0:
            tokens = self._top_p(image_features, empty_caption, top_p=top_p, temperature=temperature)
        elif top_k > 0 and top_p == 0 and alpha > 0:
            tokens = self._contrastive_search(image_features, empty_caption, top_k=top_k, alpha=alpha)

        # Decode the caption into a string.
        caption = self.tokenizer.decode(tokens)

        # Prettify the caption (optional).
        caption = self._prettify_string(caption)

        return caption
    
    def _contrastive_search(self, image_features, caption, top_k, alpha):

        # Initialize the list of word embeddings with the embedding of the first token.
        word_embeddings_matrix = self.captioner.xformer.get_submodule("decoders.0.pose_encoding.word_embeddings")
        word_embeddings = [word_embeddings_matrix(caption[:, 0]).squeeze(0)]

        for i in range(1, self.maxlen):

            # Generate logits for the next token using the captioner model.
            logits = self.captioner(memory=image_features, tgt=caption)
            logits = logits[0, i - 1]
            probabilities = F.softmax(logits, dim=-1)

            # Filter logits to keep only the top-k most probable tokens.
            confidences, indices = probabilities.sort(descending=True)
            confidences, indices = confidences[:top_k], indices[:top_k]

            # Calculate degeneration penalty for each token.
            scores = []
            for conf, ind in zip(confidences, indices):
                degeneration_penalty = max([torch.cosine_similarity(word_embeddings_matrix(ind), word, dim=-1) for word in word_embeddings])
                scores.append((1 - alpha) * conf - alpha * degeneration_penalty)
            
            # Select the token with the highest score.
            index = scores.index(max(scores))
            next_token = indices[index].item()

            # If the token is [PAD] token choose second most probable token.
            if next_token == self.tokenizer.token_to_id("[PAD]"):
                indices = indices.tolist()
                indices.remove(index)
                next_token = indices[scores.index(max(scores))]
            
            # Update the caption with the new token.
            caption[:, i] = next_token

            # Add new token's embedding to the list of embeddings.
            word_embeddings.append(word_embeddings_matrix(torch.tensor(next_token)).squeeze(0))

            # If the token is [EOS] token stop generating.
            if next_token == self.tokenizer.token_to_id("[EOS]"):
                break
        
        return caption[0].cpu().numpy()
    

    def _top_p(self, image_features, caption, top_p, temperature):
        """
        Generate a caption using top-p sampling strategy (also known as nucleus sampling or softmax sampling).

        Args:
            image_features (torch.Tensor): Image features extracted using the Vision Transformer model.
            caption (torch.Tensor): Partial caption to start the generation process.
            top_p (float): The cumulative probability threshold for top-p sampling.
            temperature (float): The temperature value for controlling token sampling randomness.

        Returns:
            numpy.ndarray: The generated caption tokens represented as a numpy array.
        """
        for i in range(1, self.maxlen):

            # Generate logits for the next token using the captioner model.
            logits = self.captioner(memory=image_features, tgt=caption)
            logits = logits[0, i - 1] / temperature
            probabilities = F.softmax(logits, dim=-1)

            # Select top-p tokens based on cumulative probabilities.
            sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
            sorted_indices_to_remove = cumulative_probabilities > top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            sorted_probabilities[sorted_indices_to_remove] = 0
            sorted_probabilities = sorted_probabilities / torch.sum(sorted_probabilities)

            # Sample a token from the top-p distribution.
            next_token_index = torch.multinomial(sorted_probabilities, num_samples=1)
            next_token = sorted_indices[next_token_index]

            # Update the caption with the new token.
            caption[:, i] = next_token.item()

            # Stop generating if the [EOS] token is selected.
            if next_token.item() == self.tokenizer.token_to_id("[EOS]"):
                break

        return caption[0].cpu().numpy()
    

    def _top_k(self, image_features, caption, top_k, temperature):
        """
        Generate a caption using top-k sampling strategy.

        Args:
            image_features (torch.Tensor): Image features extracted using the Vision Transformer model.
            caption (torch.Tensor): Partial caption to start the generation process.
            top_k (int): The number of top-k tokens to sample from.
            temperature (float): The temperature value for controlling token sampling randomness.

        Returns:
            numpy.ndarray: The generated caption tokens represented as a numpy array.
        """
        for i in range(1, self.maxlen):

            # Generate logits for the next token using the captioner model.
            logits = self.captioner(memory=image_features, tgt=caption)
            logits = logits[0, i - 1] / temperature
            probabilities = F.softmax(logits, dim=-1)

            # Select top-k tokens based on probabilities.
            top_k_values, top_k_indices = probabilities.topk(top_k)
            top_k_values = top_k_values / torch.sum(top_k_values)

            # Sample a token from the top-k distribution.
            next_token_index = torch.multinomial(top_k_values, num_samples=1)
            next_token = top_k_indices[next_token_index]

            # Update the caption with the new token.
            caption[:, i] = next_token.item()

            # Stop generating if the [EOS] token is selected.
            if next_token.item() == self.tokenizer.token_to_id("[EOS]"):
                break

        return caption[0].cpu().numpy() 
            

    def _beam_search(self, image_features, caption, beam_size):

        # Initialize beam candidates.
        beam_candidates = [{'caption': caption, 'score': 0.0}]

        for i in range(1, self.maxlen):

            # Initialize the list of new beam candidates.
            next_beam_candidates = []

            # Expand each beam candidate.
            for candidate in beam_candidates:
                partial_caption = candidate['caption']

                # Generate logits for the next token using the captioner model.
                logits = self.captioner(memory=image_features, tgt=partial_caption)
                probabilities = F.softmax(logits[0, i - 1], dim=-1)

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

        return best_caption[0].cpu().numpy()


    @staticmethod
    def _prettify_string(string):
        """
        Make the given string prettier by applying some formatting.

        Args:
            string (str): The string to prettify.

        Returns:
            str: The prettified string.
        """
        string = string.replace(" ' ", "'")
        return string


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