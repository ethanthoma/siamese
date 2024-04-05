from __future__ import absolute_import

import os
from got10k.datasets import VOT

from siamfc import TrackerSiamFC


if __name__ == "__main__":
    root_dir = os.environ.get("DATASET_PATH", "./data/VOT2018/")
    seqs = VOT(root_dir, version=2018, download=False, return_meta=True)

    tracker = TrackerSiamFC()
    tracker.train_over(seqs)
