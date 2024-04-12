from __future__ import absolute_import

import os
from got10k.experiments import ExperimentVOT

from siamfc import TrackerSiamFC
from siamfc import RecurrentAlexMul

print("TESTING")


if __name__ == "__main__":
    print("Loading model", flush=True)
    net_path = "./pretrained/v7/recurrent/siamfc_alexnet_e50.pth"
    tracker = TrackerSiamFC(net_path=net_path, model=RecurrentAlexMul)

    root_dir = os.path.expanduser("./data/VOT2018/")
    print("starting test", flush=True)
    e = ExperimentVOT(root_dir, version=2018, experiments="supervised")
    print("test initialized", flush=True)
    e.run(tracker)
    e.report([tracker.name])
