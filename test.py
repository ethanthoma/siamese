from __future__ import absolute_import

import os
from got10k.experiments import *

from siamfc import TrackerSiamFC


if __name__ == "__main__":
    net_path = "./pretrained/v2/recurrent/siamfc_alexnet_e50.pth"
    tracker = TrackerSiamFC(net_path=net_path)

    root_dir = os.path.expanduser("./data/VOT2018/")
    e = ExperimentVOT(root_dir, version=2018)
    e.run(tracker)
    e.report([tracker.name])
