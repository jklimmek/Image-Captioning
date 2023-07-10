import yaml


class Color:
    """
    A class that defines color codes for printing colored text in the terminal.
    """
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'


def yaml_to_dict(filename):
    """
    Reads a YAML file and returns its values as keyword arguments (kwargs).

    Args:
        filename (str): The path to the YAML file.

    Returns:
        dict: A dictionary containing the YAML values as keyword arguments.
    """
    with open(filename, "r") as file:
        data = yaml.safe_load(file)

    kwargs = {}
    for key, value in data.items():
        kwargs[key] = value

    return kwargs


def get_name(model, **kwargs):
    """
    Returns the name of a model.

    Args:
        model (pl.LightningModule): The model.

    Returns:
        str: The name of the model.
    """

    # Map the parameters to their abbreviations.
    param_mapping = {
        "seq_len": "S",
        "vocab_size": "V",
        "proj_dim": "P",
        "intermidiate_dim": "I",
        "embed_dim": "E",
        "nhead": "H",
        "hidden_layer_multiplier": "M",
        "activation": "A",
        "num_layers": "L",
        "dropout": "D",
        "learning_rate": "LR",
        "betas": "B",
        "weight_decay": "WD",
        "batch_size": "BS",
        "max_epochs": "ME",
    }
    # Get the number of parameters in millions.
    param_count = sum(p.numel() for p in model.parameters())
    digits = round(param_count / 1_000_000)

    # Get the parameters.
    params = {}
    for kwarg, value in kwargs.items():
        if kwarg in param_mapping:
            params[param_mapping[kwarg]] = value

    # Create the name.
    name = f"M{digits}_" + "_".join([f"{key}{value}" for key, value in params.items()])
    return name