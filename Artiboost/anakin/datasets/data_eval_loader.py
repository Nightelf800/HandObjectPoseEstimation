#-*-coding:utf-8-*-
import numpy as np
import os
import pickle

import torch
from anakin.utils.logger import logger
from anakin.utils.etqdm import etqdm
from collections import defaultdict

class HO3DData:
    def __init__(self, **cfg):
        # ======== HO3D default >>>>>>>>>>>>>>>>>
        self.raw_size = (640, 480)
        self.reorder_idxs = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20])
        # this camera extrinsic has no translation
        # and this is the reason transforms in following code just use rotation part
        self.cam_extr = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]).astype(np.float32)
        self.name = "HO3D"
        self.data_split = cfg["DATA_SPLIT"]
        self.data_root = cfg["DATA_ROOT"]
        self.root = os.path.join(self.data_root, self.name)

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        self.load_dataset()

    def load_dataset(self):
        if self.split_mode == "paper":
            seq_frames, subfolder = self._load_seq_frames()
            logger.info(f"{self.name} {self.data_split} set has frames {len(seq_frames)}")
        else:
            raise NotImplementedError()
        annot_mapping, seq_idx = self._load_annots(seq_frames=seq_frames,
                                                   subfolder=subfolder)

        annotations = {"seq_idx": seq_idx, "annot_mapping": annot_mapping}
        self.seq_idx = annotations["seq_idx"]
        self.annot_mapping = annotations["annot_mapping"]
        self.sample_idxs = list(range(len(self.seq_idx)))


    def _load_seq_frames(self, subfolder=None, seqs=None, trainval_idx=6000):
        if self.split_mode == "paper":
            if self.data_split in ["train", "trainval", "val"]:
                info_path = os.path.join(self.root, "train.txt")
                subfolder = "train"
            elif self.data_split == "test":
                info_path = os.path.join(self.root, "evaluation.txt")
                subfolder = "evaluation"
            else:
                assert False
            with open(info_path, "r") as f:
                lines = f.readlines()
            seq_frames = [line.strip().split("/") for line in lines]
        else:
            assert False
        return seq_frames, subfolder

    def _load_annots(self, seq_frames=None, subfolder="train"):
        if seq_frames is None:
            seq_frames = []
        seq_idx = []
        annot_mapping = defaultdict(list)
        seq_counts = defaultdict(int)
        annot_save = dict()
        for idx_count, (seq, frame_idx) in enumerate(etqdm(seq_frames)):
            seq_folder = os.path.join(self.root, subfolder, seq)
            meta_folder = os.path.join(seq_folder, "meta")
            rgb_folder = os.path.join(seq_folder, "rgb")

            meta_path = os.path.join(meta_folder, f"{frame_idx}.pkl")


            with open(meta_path, "rb") as p_f:
                annot_load = pickle.load(p_f)
            if annot_load["handJoints3D"].size == 3:
                annot_save["camMat"] = annot_load["camMat"]
                #annot_save["handBoundingBox"] = annot_load["handBoundingBox"]
                annot_save["handTrans"] = annot_load["handJoints3D"]
                annot_save["handJoints3D"] = annot_load["handJoints3D"][np.newaxis, :].repeat(21, 0)
                annot_save["handPose"] = np.zeros(48, dtype=np.float32)
                annot_save["handBeta"] = np.zeros(10, dtype=np.float32)

            img_path = os.path.join(rgb_folder, f"{frame_idx}.png")
            annot_save["img"] = img_path
            annot_save["frame_idx"] = frame_idx

            annot_mapping[seq].append(annot_save)
            seq_idx.append((seq, seq_counts[seq]))
            seq_counts[seq] += 1

        return annot_mapping, seq_idx