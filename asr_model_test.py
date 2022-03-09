'''Utility code for loading ASR models'''
from deepspeech_main.deepspeech_pytorch.model import DeepSpeech
from deepspeech_main.deepspeech_pytorch.utils import load_model
import torch
from art.estimators.speech_recognition import PyTorchDeepSpeech
from armory.data.utils import maybe_download_weights_from_s3
from deepspeech_main.deepspeech_pytorch.configs.train_config import BiDirectionalConfig, OptimConfig, SpectConfig
import pickle
from pdb import set_trace as bp
from omegaconf.dictconfig import DictConfig

LABELS = ['_', "'", 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', ' ']
PRECISION = 16
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def get_art_model(model_kwargs, wrapper_kwargs, weights_file=None):
    weights_file_path = maybe_download_weights_from_s3(weights_file)
    model = DeepSpeech( # Create generic Deepspeech instace to load weights into
        labels = LABELS,
        model_cfg= BiDirectionalConfig(),
        precision = PRECISION,
        optim_cfg = OptimConfig(),
        spect_cfg = DictConfig(SpectConfig()),
        **model_kwargs
    )
    ckpt = torch.load(weights_file_path)
    model.load_state_dict(ckpt["state_dict"])
    #model = torch.load(weights_file_path)
    #model = DeepSpeech() #Create generic Deepspeech instace to load weights into
    #state_dict = torch.load(weights_file_path)["state_dict"]
    #model.load_state_dict(state_dict)
    wrapped_model = PyTorchDeepSpeech( model=model, **wrapper_kwargs )
    return wrapped_model
