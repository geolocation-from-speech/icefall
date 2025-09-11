# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from lhotse import CutSet, Fbank, FbankConfig, load_manifest, load_manifest_lazy
from lhotse.dataset import (
    CutConcatenate,
    CutMix,
    DynamicBucketingSampler,
    PrecomputedFeatures,
    SpecAugment,
)
from dataset import K2MultiTalkerSpeechRecognitionDataset 
from lhotse.dataset.sampling.cut_splice import CutSpliceIterable
from lhotse.utils import fix_random_seed
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from icefall.utils import str2bool


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriSpeechAsrDataModule:
    """
    DataModule for k2 ASR experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).
    It contains all the common data pipeline modules used in ASR
    experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction
    This class should be derived for specific corpora used in ASR tasks.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="ASR data related options",
            description="These options are used for the preparation of "
            "PyTorch DataLoaders from Lhotse CutSet's -- they control the "
            "effective batch sizes, sampling strategies, applied data "
            "augmentations, etc.",
        )
        group.add_argument(
            "--manifest-dir",
            type=Path,
            default=Path("data/manifests"),
            help="Path to directory with train/valid/test cuts.",
        )
        group.add_argument(
            "--enable-musan",
            type=str2bool,
            default=True,
            help="When enabled, select noise from MUSAN and mix it "
            "with training dataset. ",
        )
        group.add_argument(
            "--concatenate-cuts",
            type=str2bool,
            default=False,
            help="When enabled, utterances (cuts) will be concatenated "
            "to minimize the amount of padding.",
        )
        group.add_argument(
            "--duration-factor",
            type=float,
            default=1.0,
            help="Determines the maximum duration of a concatenated cut "
            "relative to the duration of the longest cut in a batch.",
        )
        group.add_argument(
            "--gap",
            type=float,
            default=1.0,
            help="The amount of padding (in seconds) inserted between "
            "concatenated cuts. This padding is filled with noise when "
            "noise augmentation is used.",
        )
        group.add_argument(
            "--max-duration",
            type=int,
            default=100.0,
            help="Maximum pooled recordings duration (seconds) in a "
            "single batch. You can reduce it if it causes CUDA OOM.",
        )
        group.add_argument(
            "--num-buckets",
            type=int,
            default=100,
            help="The number of buckets for the BucketingSampler"
            "(you might want to increase it for larger datasets).",
        )
        group.add_argument(
            "--shuffle",
            type=str2bool,
            default=True,
            help="When enabled (=default), the examples will be "
            "shuffled for each epoch.",
        )

        group.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="The number of training dataloader workers that "
            "collect the batches.",
        )
        group.add_argument(
            "--enable-spec-aug",
            type=str2bool,
            default=True,
            help="When enabled, use SpecAugment for training dataset.",
        )
        group.add_argument(
            "--spec-aug-time-warp-factor",
            type=int,
            default=80,
            help="Used only when --enable-spec-aug is True. "
            "It specifies the factor for time warping in SpecAugment. "
            "Larger values mean more warping. "
            "A value less than 1 means to disable time warp.",
        )
        group.add_argument(
            "--normalize-volume",
            type=str2bool,
            default=False,
            help="Normalize the volume of each cutsplice segment",
        )

    def train_dataloaders(
        self,
        cuts_train: CutSet,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          cuts_train:
            CutSet for training.
          sampler_state_dict:
            The state dict for the training sampler.
        """
        input_transforms = []
        logging.info("About to create train dataset")
        train = K2MultiTalkerSpeechRecognitionDataset(
            cut_transforms=[],
            input_transforms=input_transforms,
        )

        logging.info("Using DynamicBucketingSampler.")
        train_sampler = DynamicBucketingSampler(
            cuts_train,
            max_duration=self.args.max_duration,
            shuffle=False,
            num_buckets=self.args.num_buckets,
            drop_last=True,
            quadratic_duration=20,
        )
        logging.info("About to create train dataloader")
        
        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloaders(self, cuts_valid: CutSet) -> DataLoader:
        validate = K2MultiTalkerSpeechRecognitionDataset(
            return_cuts = self.args.return_cuts,
            cut_transforms=[],
        )
        
        valid_sampler = DynamicBucketingSampler(
            cuts_valid,
            max_duration=self.args.max_duration,
            shuffle=False,
        )
        
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloaders(self, cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = K2MultiTalkerSpeechRecognitionDataset(
            return_cuts=self.args.return_cuts
        )
        sampler = DynamicBucketingSampler(
            cuts, max_duration=self.args.max_duration, shuffle=False
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def train_cuts(self) -> CutSet:
        logging.info("About to get train cuts")
        path = "/ocean/projects/cis210027p/scornell/mdctc/icefall/egs/librispeech/MTASR/manifests/sim-librispeech-train-cutset.jsonl.gz"
        speakers = list(Path("data/manifests/librispeech_speakers").rglob("*.jsonl.gz"))
        num_spks = len(speakers) 
        weight = 1/num_spks
        sim_weight = 0.0
        cut_info = []
        for s in speakers:
            cut_info.append((s.stem.split("_")[-1].split(".")[0], (1-sim_weight)*weight, s))
        
        cutsets, weights, names = [], [], []
        for n, w, p in cut_info: 
            cutsets.append(load_manifest_lazy(p))
            weights.append(w)
            names.append(n)

        #cut_info.append(("sim", sim_weight, path))
        #cutsets.append(load_manifest_lazy(path))
        #weights.append(sim_weight)
        #names.append("sim")
        #recordings = list(Path(path).rglob("*train*.jsonl.gz"))
        #num_recos = len(recordings)
        #weight = 1/num_recos
        #cut_info = []
        #for s in recordings:
        #    cut_info.append((s.stem.split("_")[-1].split(".")[0], weight, s))
        #
        #cutsets, weights, names = [], [], []
        #for n, w, p in cut_info: 
        #    cutsets.append(load_manifest_lazy(p))
        #    weights.append(w)
        #    names.append(n)

        cs_iter = CutSpliceIterable(
            cutsets,
            cutset_weights=weights,
            cutset_prefixes=names,
            max_duration=20,
            final_max_duration=30,
            max_duration_increment=1e-03,
            max_num_overlaps=2,
            max_splices=2,
            min_splices=2,
            final_max_splices=3,
            final_min_splices=2,
            max_splices_schedule_increment=8e-05,
            min_splices_schedule_increment=8e-05,
            max_unique=10,
            max_overlap=[0.95]*num_spks,
            min_overlap=[0.5]*num_spks,
            max_snr=[60 * (random.random() - 0.5) for i in range(num_spks)],
            normalize_loudness=False,
            serialize='none',
            sampling_rate=16000,
        )
        return CutSet(cs_iter)

    @lru_cache()
    def libricss_cuts(self) -> CutSet:
        logging.info("About to get libricss cuts")
        cuts = load_manifest_lazy("data/manifests/libricss-ihm-mix_segments_all.jsonl.gz")
        cuts = cuts.filter(lambda c: c.id.split("_")[1].split("-")[0] != "session0")
        return cuts

    @lru_cache()
    def single_speaker_cuts(self) -> CutSet:
        test_other = load_manifest_lazy("./data/manifests/cuts_librispeech_test-other.jsonl.gz")
        test_clean = load_manifest_lazy("./data/manifests/cuts_librispeech_test-clean.jsonl.gz")
        return test_other, test_clean

    @lru_cache()
    def libri2mix_dev_clean_cuts(self) -> CutSet:
        cuts = load_manifest_lazy("./data/manifests/libri2mix_mix_clean_sc_dev_cutset.jsonl.gz")
        return cuts

    @lru_cache()
    def libri2mix_test_clean_cuts(self) -> CutSet:
        cuts = load_manifest_lazy("./data/manifests/libri2mix_mix_clean_sc_test_cutset.jsonl.gz")
        return cuts

    @lru_cache()
    def libri2mix_test_both_cuts(self) -> CutSet:
        cuts = load_manifest_lazy("./data/manifests/libri2mix_mix_both_sc_test_cutset.jsonl.gz")
        return cuts

    @lru_cache()
    def libri3mix_test_clean_cuts(self) -> CutSet:
        cuts = load_manifest_lazy("./data/manifests/libri3mix_mix_clean_sc_test_cutset.jsonl.gz")
        return cuts


    @lru_cache()
    def valid_cuts(self) -> CutSet:
        logging.info("About to get dev cuts")
        cut_info = [
            ('dev-clean', 0.5, "./data/manifests/cuts_librispeech_dev-clean.jsonl.gz"),
            ('dev-other', 0.5, "./data/manifests/cuts_librispeech_dev-other.jsonl.gz"),
        ]
        cutsets, weights, names = [], [], []
        for n, w, p in cut_info: 
            cutsets.append(load_manifest_lazy(p))
            weights.append(w)
            names.append(n)

        cs_iter = CutSpliceIterable(
            cutsets,
            cutset_weights=weights,
            cutset_prefixes=names,
            max_duration=30,
            max_splices=2,
            min_splices=2,
            max_overlap=[0.95, 0.95],
            min_overlap=[0.8, 0.8],
            max_snr=[0, 0],
            final_max_splices=2,
            max_unique=2,
            normalize_loudness=False,
            serialize='none',
            sampling_rate=16000,
        )
        
        # Just do two hours of data
        total_duration = 0
        cuts = []
        for c in cs_iter:
            if total_duration > 2*3600:
                break
            total_duration += c.duration
            cuts.append(c)     
        return CutSet(cuts)

    
    #@lru_cache()
    #def valid_cuts(self) -> CutSet:
    #    logging.info("About to get dev cuts")
    #    recordings = list(Path("data/manifests/ami_recordings_dev").rglob("*.jsonl.gz"))
    #    num_recos = len(recordings)
    #    weight = 1/num_recos
    #    cut_info = []
    #    for s in recordings:
    #        cut_info.append((s.stem.split("_")[-1].split(".")[0], weight, s))
    #    
    #    cutsets, weights, names = [], [], []
    #    for n, w, p in cut_info: 
    #        cutsets.append(load_manifest_lazy(p))
    #        weights.append(w)
    #        names.append(n)

    #    cs_iter = CutSpliceIterable(
    #        cutsets,
    #        cutset_weights=weights,
    #        cutset_prefixes=names,
    #        max_duration=30,
    #        max_splices=2,
    #        min_splices=2,
    #        final_max_splices=2,
    #        final_min_splices=2,
    #        max_splices_schedule_increment=4e-05,
    #        min_splices_schedule_increment=4e-05,
    #        max_unique=2,
    #        max_overlap=[0.0]*num_recos,
    #        min_overlap=[0.0]*num_recos,
    #        max_snr=[60 * (random.random() - 0.5) for i in range(num_recos)],
    #        normalize_loudness=False,
    #        serialize='none',
    #        sampling_rate=16000,
    #    )
    #    
    #    # Just do two hours of data
    #    total_duration = 0
    #    cuts = []
    #    for c in cs_iter:
    #        if total_duration > 2*3600:
    #            break
    #        total_duration += c.duration
    #        cuts.append(c)     
    #    return CutSet(cuts)

    @lru_cache()
    def synth_cuts(self) -> CutSet:
        logging.info("About to get snythetic cuts")
        return load_manifest_lazy("data/manifests/cuts_librispeech_dev_synth.jsonl.gz")        

    @lru_cache()
    def aed_cuts(self) -> CutSet:
        logging.info("About to get audioset cuts")
        return load_manifest_lazy("data/manifests/audioset_eval_whole/cuts_0_mapped.jsonl.gz").subset(first=100) 

    @lru_cache()
    def ami_dev(self) -> CutSet:
        logging.info("About to get ami dev cuts")
        return load_manifest_lazy("data/manifests/cuts_ami_dev_pause0.5.jsonl.gz")
   
    @lru_cache()
    def ami_test(self) -> CutSet:
        logging.info("About to get ami test cuts")
        return load_manifest_lazy("data/manifests/cuts_ami_test_pause0.5.jsonl.gz")


def test():
    parser = argparse.ArgumentParser()
    Scale23SpeechAsrDataModule.add_arguments(parser)
    args = parser.parse_args()
    adm = Scale23SpeechAsrDataModule(args)

    cuts = adm.train_cuts()
    dl = adm.train_dataloaders(cuts)
    for i, batch in tqdm(enumerate(dl)):
        if i == 100:
            break

    cuts = adm.dev_cuts()
    dl = adm.valid_dataloaders(cuts)
    for i, batch in tqdm(enumerate(dl)):
        if i == 100:
            break


if __name__ == "__main__":
    test()
