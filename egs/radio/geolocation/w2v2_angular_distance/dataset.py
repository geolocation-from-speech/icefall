from functools import reduce
from lhotse import validate
from lhotse.cut import CutSet
from lhotse.dataset.collation import collate_audio
from lhotse.dataset.input_strategies import BatchIO, AudioSamples
from lhotse.workarounds import Hdf5MemoryIssueFix
from lhotse.utils import ifnone

import torch
from typing import Callable, Dict, Union, List


class GeolocationDataset(torch.utils.data.Dataset):
    """
    Dataset that contains supervisions for geoloation, i.e., just the lat, lon
    coordinates. Depending on which data sets are used, it could be useful
    to return some other attributes as well. Checks for this are performed
    automatically and any applicable attributes in the manifest are included,
    for instance, speaker or language. These fields can be useful in LID
    experiments.

    This dataset assumes that there are cuts with a custom field in the
    supervisions which itself has field "lat", and "lon". In other words, to
    access the location of an audio signal, whatever that location means, it
    would be 
    .. code-block::

        {
            'lat': cuts[0].supervisions[0].custom['lat'],
            'lon': cuts[0].supervisions[0].custom['lon'],
        }

    The coordinates will be returned as a T = B x 2 tensor, where
    T[0][0] would be the latitude of the 0th element in the minibatch.
    """
    def __init__(
        self, 
        return_cuts: bool = False,
        use_feats: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = None,
        input_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = None,
        input_strategy: BatchIO = AudioSamples(),
        reverse: float = 0.0,
    ):
        super().__init__()
        self.return_cuts = return_cuts
        self.use_feats = use_feats
        self.cut_transforms = ifnone(cut_transforms, [])
        self.input_transforms = ifnone(input_transforms, [])
        self.input_strategy = input_strategy
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=50)

        # This attribute is mostly for diagnostic purposes when figuring out
        # to what extent a model is using sequential information at inference
        # time. It probably could be used as a data perturbation for
        # sequence-level classification tasks, but we have not tested that yet.
        self.reverse = reverse

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        self._validate(cuts)
        self.hdf5_fix.update()
        cuts = cuts.sort_by_duration(ascending=False)
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)
        cuts = cuts.sort_by_duration(ascending=False)
        if self.reverse > 0.0:
            new_cuts = []
            for c in cuts:
                new_cuts.append(
                    reduce(
                        lambda a, b: a.append(b),
                        list(c.cut_into_windows(self.reverse))[::-1]
                    )
                )
            cuts = CutSet.from_cuts(new_cuts)
        inputs, input_lens = collate_audio(cuts)
        # Get a tensor with batched feature matrices, shape (B, T, F)
        # Collation performs auto-padding, if necessary.
        supervision_intervals = self.input_strategy.supervision_intervals(cuts)
        
        segments = torch.stack(list(supervision_intervals.values()), dim=1)
        for tnfm in self.input_transforms:
            inputs = tnfm(inputs, supervision_segments=segments)

        if cuts[0].supervisions[0].custom is not None:
            targets = torch.Tensor(
                [
                    [s.custom['lat'], s.custom['lon']]
                    for c in cuts for s in c.supervisions
                ]
            )
        else:
            targets = None
      
        if cuts[0].supervisions[0].speaker is not None:
            spks = [c.supervisions[0].speaker for c in cuts]
        else:
            spks = None

        batch = {
            "supervisions": {
                "targets": targets,
                "language": [c.supervisions[0].language for c in cuts],
                "spks": spks,
            },
            "ids": [c.id for c in cuts],
            "inputs": inputs,
            "features_lens": input_lens,
        }
        return batch

    def _validate(self, cuts: CutSet) -> None:
        validate(cuts)

