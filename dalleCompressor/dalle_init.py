import torch

from utils import TrainDecoderConfig


def decoder_init(decoder_path="weights/second_decoder"):
    decoder_config = TrainDecoderConfig.from_json_path(f"{decoder_path}.json").decoder
    decoder_config.sample_timesteps = 500
    decoder_config.ddim_sampling_eta = 0
    decoder = decoder_config.create()
    decoder_model_state = torch.load(f"{decoder_path}.pth", map_location=torch.device('cuda'))
    decoder.load_state_dict(decoder_model_state, strict=False)
    print("Decoder is initialized")
    return decoder.cuda()
