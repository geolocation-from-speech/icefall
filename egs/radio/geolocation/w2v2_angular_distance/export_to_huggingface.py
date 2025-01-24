import argparse
from transformers import Wav2Vec2Model
from pathlib import Path
from icefall.utils import AttributeDict 
from icefall.env import get_env_info
import torch
import torch.nn as nn



def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="zipformer/exp",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint-100000",
    )

    parser.add_argument(
        "--hf-name",
        type=str,
        default="geolocation-big-10s"
    )
    
    return parser


def get_model(params: AttributeDict) -> nn.Module:
    with open(Path(params.exp_dir) / "args.conf", 'r') as f:
        conf = eval(f.readline())
    
    model = Wav2Vec2Model(
        modelpath=conf['modelpath'],
        pooling_loc=conf['pooling_loc'],
        pooling_type=conf['pooling_type'],
        freeze_feat_extractor = False, # Possibly necessary for huggingface export
    ) 
    return model


def get_params() -> AttributeDict:
    params = AttributeDict(
        {
            "best_train_loss": float("inf"),
            "best_valid_loss": float("inf"),
            "best_train_epoch": -1,
            "best_valid_epoch": -1,
            "batch_idx_train": 0,
            "log_interval": 1, #25
            "reset_interval": 200,
            "valid_interval": 10,  # 500
            # parameters for zipformer
            "env_info": get_env_info(),
        }
    )
    return params


def export_to_hf(args):
    params = get_params()
    params.update(vars(args))

    mdl_dict = torch.load(
        f"{params.exp_dir}/{params.checkpoint}.pt",
        map_location="cpu"
    )['model']

    with open(Path(params.exp_dir) / "args.conf", 'r') as f:
        conf = eval(f.readline())
    
    mdl = Wav2Vec2Model.from_pretrained(conf['modelpath'])
    mdl2_dict = mdl.state_dict()
    for k, p in mdl_dict.items():
        if k.startswith("encoder.encoder"):
            k_ = '.'.join(k.split(".")[1:])
            mdl2_dict[k_] = p.data

    mdl.load_state_dict(mdl2_dict)

    # Make the huggingface export dir
    hf_export = Path(f"{params.exp_dir}/huggingface_export_{params.checkpoint}")
    hf_export.mkdir(mode=511, parents=True, exist_ok=True)
    mdl.save_pretrained(hf_export)
    mdl.push_to_hub(args.hf_name)

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)
    export_to_hf(args)


if __name__ == "__main__":
    main()
