from __future__ import absolute_import

import os
from got10k.experiments import ExperimentGOT10k

from siamfc import TrackerSiamFC
from siamfc import RecurrentAlexMul


if __name__ == "__main__":
    net_path = "./pretrained/v7/recurrent/siamfc_alexnet_e10.pth"
    tracker = TrackerSiamFC(net_path=net_path, model=RecurrentAlexMul)

    root_dir = os.environ.get("DATASET_PATH", "./data/VOT2018/")
    e = ExperimentGOT10k(root_dir, subset="test")
    e.run(tracker)
    e.report([tracker.name])
