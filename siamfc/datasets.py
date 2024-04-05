from __future__ import absolute_import, division

import numpy as np
import cv2
from torch.utils.data import Dataset

__all__ = ["Pair"]


class Pair(Dataset):
    def __init__(self, seqs, transforms=None, pairs_per_seq=1):
        super(Pair, self).__init__()

        print("Dataset Class is initialized for the First Time :)")

        self.seqs = seqs
        self.transforms = transforms
        self.pairs_per_seq = pairs_per_seq
        # self.indices = np.random.permutation(len(seqs))
        self.indices = np.arange(len(seqs))

        """ Load Data in Sequence
        we want to load the data sequentially to make use of the RNN layer, so we need to keep track of the seq frames
        and make sure they are loaded in seq. :) """
        self.seq_index = 0  # to keep track of seq index
        self.frame_index = 0  # to keep track of the frame index in the seq
        self.reset_hidden = False  # keep track of when to reset hidden state

        self.return_meta = getattr(seqs, "return_meta", False)

    def __getitem__(self, index):
        # index = self.indices[index % len(self.indices)]
        if self.frame_index >= len(self.seqs[self.seq_index][0]) - 1:
            self.seq_index = self.seq_index + 1  # if this seq is done, go to next one
            self.frame_index = 0  # reset the frame index to 0

        # get filename lists and annotations
        if self.return_meta:
            img_files, anno, meta = self.seqs[self.seq_index]
            vis_ratios = meta.get("cover", None)
        else:
            img_files, anno = self.seqs[self.seq_index][:2]
            vis_ratios = None

        if self.frame_index == 0:
            # print("Reset hidden is set to True")
            self.reset_hidden = True
        else:
            # print("Reset hidden is set to False")
            self.reset_hidden = False

        # sample a frame pair
        rand_z, rand_x = self._sample_pair()

        z = cv2.imread(img_files[rand_z], cv2.IMREAD_COLOR)
        x = cv2.imread(img_files[rand_x], cv2.IMREAD_COLOR)
        z = cv2.cvtColor(z, cv2.COLOR_BGR2RGB)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

        box_z = anno[rand_z]
        box_x = anno[rand_x]

        item = (z, x, box_z, box_x)
        if self.transforms is not None:
            item = self.transforms(*item)

        return item, self.reset_hidden

    def __len__(self):
        return len(self.indices) * self.pairs_per_seq

    def _sample_pair(self):
        rand_z = self.frame_index
        next_frame = rand_z + 1
        rand_x = (
            next_frame
            if next_frame <= len(self.seqs[self.seq_index][0]) - 1
            else rand_z
        )
        self.frame_index = self.frame_index + 1  # increment frame index
        return rand_z, rand_x

    def _filter(self, img0, anno, vis_ratios=None):
        size = np.array(img0.shape[1::-1])[np.newaxis, :]
        areas = anno[:, 2] * anno[:, 3]

        # acceptance conditions
        c1 = areas >= 20
        c2 = np.all(anno[:, 2:] >= 20, axis=1)
        c3 = np.all(anno[:, 2:] <= 500, axis=1)
        c4 = np.all((anno[:, 2:] / size) >= 0.01, axis=1)
        c5 = np.all((anno[:, 2:] / size) <= 0.5, axis=1)
        c6 = (anno[:, 2] / np.maximum(1, anno[:, 3])) >= 0.25
        c7 = (anno[:, 2] / np.maximum(1, anno[:, 3])) <= 4
        if vis_ratios is not None:
            c8 = vis_ratios > max(1, vis_ratios.max() * 0.3)
        else:
            c8 = np.ones_like(c1)

        mask = np.logical_and.reduce((c1, c2, c3, c4, c5, c6, c7, c8))
        val_indices = np.where(mask)[0]

        return val_indices
