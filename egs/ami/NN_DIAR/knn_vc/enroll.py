import argparse
from lhotse import load_manifest_lazy
from model import EncoderWrapper
import faiss
from pathlib import Path
import torch


def main(args):
    # Take care of the directory where the profiles will be saved
    enroll_dir = Path(args.enroll_dir)
    enroll_dir.mkdir(parents=True, exist_ok=True)
    
    # Create the encoder which will be used for profile generation (enrollment)
    encoder = EncoderWrapper(
        ssl_src=args.ssl_src,
        layer=args.wavlm_layer,
    )
    
    # Load the lhotse cutset
    cuts = load_manifest_lazy(args.enrollment_cutset)
    if torch.cuda.is_available():
        encoder.to('cuda')
        device = 'cuda'

    # Enroll with the cuts
    index = encoder.enroll(cuts)
    # Save to disk
    faiss.write_index(index, str(enroll_dir / f"{args.index_name}.faiss"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("enrollment_cutset")
    parser.add_argument(
        "--enroll-dir",
        type=str,
        default="enrollment",
        help="The directory in which to save the enrollment index",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="speakers",
    )
    parser.add_argument(
        "--wavlm-layer",
        type=int,
        default=None,
        help="The layer at which we extract embeddings for matching"
    )
    parser.add_argument(
        "--ssl-src",
        type=str,
        choices=["torchaudio", "microsoft"],
        default="torchaudio",
        help="Which pretrained implementation of wavlm to use",
    )
    args = parser.parse_args()
    main(args)
