import os
import json
import argparse
import evaluate
import tabulate
import pandas as pd
from tqdm import tqdm
from PIL import Image
from .captioner import ImageCaptioner
from .utils import Color


def evaluate(captioner, csv_file, images_dir, beam_size, store_json=False):
    """
    Evaluates the performance of a captioning model using SacreBLEU metric.

    Args:
        captioner (object): The captioning model.
        csv_file (str): Path to the CSV file containing image captions.
        images_dir (str): Path to the directory containing images.
        beam_size (int): Beam size for caption generation.
        store_json (bool, optional): Whether to store the evaluation results as JSON. Defaults to False.

    Returns:
        float: SacreBLEU score representing the quality of generated captions.
    """

    result_dict = {}

    # Read the CSV file.
    df = pd.read_csv(csv_file, sep="|")
    groups = df.groupby(by="image_name")

    # Generate captions for each image.
    for image_name, vals in tqdm(groups, total=len(groups), ncols=100):
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        caption = captioner.generate_caption(image, beam_size=beam_size)
        result_dict[image_name] = {}
        result_dict[image_name]["hyps"] = [caption]
        result_dict[image_name]["refs"] = vals["comment"].tolist()[:5]

    # Store results as JSON if required.
    if store_json is True:
        with open("results.json", "w") as f:
            json.dump(result_dict, f)

    # Prepare caption and reference lists for SacreBLEU computation.
    hyps, refs = [], []
    for item in result_dict.values():
        hyps.append(item["hyps"][0].capitalize().rstrip("."))
        refs.append([ref.rstrip(".") for ref in item["refs"]])

    # Compute SacreBLEU score.
    bleu = evaluate.load("sacrebleu")
    sacrebleu_score = bleu.compute(predictions=hyps, references=refs)
    return sacrebleu_score["score"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the captioning model images.")
    parser.add_argument("--checkpoint", type=str, required=True, help="The path to the model checkpoint file.")
    parser.add_argument("--tokenizer", type=str, required=True, help="The path to the tokenizer file.")
    parser.add_argument("--csv-file", type=str, required=True, help="The path to the CSV with columns: 'image_name', 'caption'.")
    parser.add_argument("--images-dir", type=str, required=True, help="The path to the directory containing the images.")
    parser.add_argument("--beam-size", type=int, default=1, help="The beam size to use for beam search.")
    parser.add_argument("--store_json", action="store_true", help="Store the results in a JSON file.")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA.")
    parser.parse_args()

    # Set the device.
    device = "cuda" if parser.cuda else "cpu"

    # Load the captioner.
    captioner = ImageCaptioner.from_files(
        checkpoint_path=parser.checkpoint,
        tokenizer_path=parser.tokenizer,
        device=device
    )

    # Evaluate the model.
    sacrebleu = evaluate(
        captioner=captioner, 
        csv_file=parser.csv_file, 
        images_dir=parser.images_dir, 
        beam_size=parser.beam_size,
        store_json=parser.store_json
    )

    # Create table (More metrics can be added here).
    table = [
        [f"{Color.BLUE}BLEU{Color.ENDC}", sacrebleu], 
    ]

    # Print scores.
    print("\n", tabulate(table, headers=["Metric", "Score"], floatfmt=".2f"))