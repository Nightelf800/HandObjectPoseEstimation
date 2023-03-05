import torch
import os
import random
import numpy as np
import time
from anakin.utils.logger import logger
from anakin.opt import arg, cfg
from anakin.submit import SubmitEpochPass
from anakin.utils import builder
from anakin.models.arch import Arch


def set_all_seeds(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main_work(gpu_id, time_f):
    rank = 0
    set_all_seeds(cfg["TRAIN"]["MANUAL_SEED"])
    submit_picture = SubmitEpochPass.build(arg.submit_dataset, cfg=None)

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

    logger.info("--------------------------Save_path----------------------------")

    root_path = "./exp"
    output_file_path = "points_data.txt"
    time_stamp = time.strftime("%Y_%m%d_%H%M_%S", time.localtime(time.time()))
    dump_path = os.path.join(root_path, f"submit_picture_{time_stamp}")
    if not os.path.exists(dump_path):
        os.makedirs(dump_path)
    dump_path = os.path.join(dump_path, "_", output_file_path)

    logger.info("----------------------------Test------------------------------")

    with torch.no_grad():
        model.eval()
        submit_picture(epoch_idx=0,
                       data_loader=test_loader,
                       arch_model=model,
                       rank=rank,
                       dump_path=dump_path)

    logger.info("---------------------------Complete-----------------------------")




def main():
    exp_time = time()
    logger.info("======> Start Execute Algorithm <======")
    main_work(arg.gpus[0], exp_time)


if __name__ == '__main__':
    main()