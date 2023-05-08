import torch
import os
import random
import numpy as np
import time
import json
from typing import (Any, Callable, List, Mapping, Optional, Sequence, TypeVar, Union)
from anakin.utils.logger import logger
from anakin.datasets.hodata import ho_collate
from anakin.opt import arg, cfg
from anakin.submit import SubmitEpochPass
from anakin.utils import builder
from anakin.utils.etqdm import etqdm
from anakin.models.arch import Arch
from anakin.criterions.criterion import Criterion


def set_all_seeds(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_order_idxs():
    reorder_idxs = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]
    unorder_idxs = np.argsort(reorder_idxs)
    return reorder_idxs, unorder_idxs


def roundall(rows):
    return [[round(val, 5) for val in row] for row in rows]


def dump_json(pred_out_path, res_joints):
    xyz_pred_list = dict()
    xyz_pred_list["data_type"] = "json"
    xyz_pred_list["model_name"] = "HO3D"
    xyz_pred_list["predict"] = [roundall(x.tolist()) for x in res_joints]

    # save to a json
    with open(pred_out_path, "w") as fo:
        json.dump(xyz_pred_list, fo)
    logger.info("Dumped %d joints predictions to %s" % \
                (len(xyz_pred_list), pred_out_path))


def evaluator(
    data_loader: [torch.utils.data.DataLoader],
    arch_model: [Arch],
    rank: int
):
    arch_model.eval()

    # ? <<<<<<<<<<<<<<<<<<<<<<<<<
    res_joints: List[Any] = []
    reorder_idxs, unorder_idxs = get_order_idxs()
    # ? >>>>>>>>>>>>>>>>>>>>>>>>>

    bar = etqdm(data_loader, rank=rank)
    for batch_idx, batch in enumerate(bar):
        predict_arch_dict = arch_model(batch)
        predicts = {}
        for key in predict_arch_dict.keys():
            predicts.update(predict_arch_dict[key])

        pred_joints = predicts["joints_3d_abs"].detach()

        pred_joints[:, 0] = batch["root_joint"].to(pred_joints.device)

        pred_joints = predicts["joints_3d_abs"].cpu().detach()[:, unorder_idxs]
        pred_joints[:, :, 0] = -pred_joints[:, :, 0]
        joints = [-val.numpy()[0] for val in pred_joints.split(1)]
        res_joints.extend(joints)

    dump_path = "exp/json_data/data_h03d.json"
    dump_json(dump_path, res_joints)


def main_work():
    rank = 0
    set_all_seeds(cfg["TRAIN"]["MANUAL_SEED"])
    logger.info("-------------------------Test_data--------------------------")

    test_data = builder.build_dataset(cfg["DATASET"]["TEST"], preset_cfg=cfg["DATA_PRESET"])
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=arg.batch_size,
                                              shuffle=False,
                                              num_workers=int(arg.workers),
                                              drop_last=False,
                                              collate_fn=ho_collate)

    logger.info("---------------------------Model----------------------------")

    model_list = builder.build_arch_model_list(cfg["ARCH"], preset_cfg=cfg["DATA_PRESET"])
    model = Arch(cfg, model_list=model_list)
    model = torch.nn.DataParallel(model).to(arg.device)

    logger.info("----------------------------Test------------------------------")

    with torch.no_grad():
        model.eval()
        evaluator(data_loader=test_loader, arch_model=model, rank=rank)

    logger.info("---------------------------Complete-----------------------------")




def main():
    logger.info("======> Start Execute Algorithm <======")
    main_work()


if __name__ == '__main__':
    main()