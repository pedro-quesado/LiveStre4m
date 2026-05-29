from utils.interpolation_models.EDEN import EDEN
from utils.interpolation_models.EDEN_VAE import VAE
from utils.interpolation_models.EDEN_DiT import DiT
from utils.interpolation_models.Discriminator import NLayerDiscriminator


def load_model(model_name, **model_args):
    if model_name == "EDEN":
        return EDEN(**model_args)
    elif model_name == "EDEN_VAE":
        return VAE(**model_args)
    elif model_name == "EDEN_DiT":
        return DiT(**model_args)
    elif model_name == "Discriminator":
        return NLayerDiscriminator(**model_args)
    else:
        raise f"No model named {model_name} in models!"
