import torch
from dataloader import *
from data_eval_loader import *
from torch.utils.data import Dataset, DataLoader
from utils.utils import model_builder
from utils.catch_camera_image import *
from utils.HandposeEvaluation import *
from utils.SocketToUnity import *
import pickle
import tqdm
import sys
import os
from PIL import Image
import numpy as np
import time
import argparse
import threading
import json
import re
from utils.forwardpass import get_EvalFunction

eval_order = "--path MyDataSet/checkpoints"


def get_number(val):
    return int(re.findall('[0-9]+', val)[0])


def round_all(rows):
    return [[round(val, 5) for val in row] for row in rows]


def convey_data(estimation_xyz):
    res_xyz_list = dict()
    res_xyz_list["data_type"] = "json"
    res_xyz_list["model_name"] = "Hglass"
    res_xyz_list["predict"] = [round_all(x.tolist()) for x in estimation_xyz]

    send_data.data_save(res_xyz_list)


def hand_evaluate(data, test_set, output_to_pred):
    inputs, com, M_inv, cubesize, original = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

    outputs = model(inputs)

    end_time = time.time()  # 程序结束时间
    run_time = end_time - start_time  # 程序的运行时间，单位为秒
    print("运行时间:", run_time)

    preds = output_to_pred(inputs, outputs, cubesize, com, setting)

    with open("res/res2.json", "a+") as fo:
        res_xyz_list = dict()
        res_xyz_list["data_type"] = "json"
        res_xyz_list["model_name"] = "Hglass"
        res_xyz_list["predict"] = [round_all(x.tolist()) for x in preds]
        json.dump(res_xyz_list, fo)


    prediction_UVDorig = CropToOriginal(preds, M_inv.float())
    estimation_xyz = test_set.convert_uvd_to_xyz_tensor(prediction_UVDorig)
    convey_data(estimation_xyz)


def _async_raise(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("invalid thread id")
    elif res != 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def stop_thread(thread):
    _async_raise(thread.ident, SystemExit)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--batch_size', default=32, type=int, help="batch_size")
    parser.add_argument('--cuda_id', default=-1, type=int, help="Cuda ID")
    parser.add_argument('--path', default="", type=str, help="the address of the dataset", required=True)
    parser.add_argument('--num_workers', default=4, type=int, help="num of subprocesses for data loading")
    parser.add_argument('--joint_dim', default=3, type=int, help="determine if it is 3D or 2D")
    parser.add_argument('--dataset', default="nyu", choices=('nyu', 'icvl', 'msra'), type=str,
                        help="which dataset to use")

    args = parser.parse_args(eval_order.split())

    con_socket = threading.Condition()
    con_camera = threading.Condition()
    use_file = False

    list_files = os.listdir(args.path)
    list_files.sort(key=get_number)
    model_path = os.path.join(args.path, list_files[0])
    setting = torch.load(model_path)["args"]
    args.dataset = setting.dataset

    print("MSRA dataset will be used")
    if os.environ.get('MSRA_PATH') is None:
        os.environ["MSRA_PATH"] = "data/MSRA"

    if args.cuda_id == -1:
        default_cuda_id = "cuda:{}".format(int(setting.default_cuda_id))
    else:
        default_cuda_id = "cuda:{}".format(args.cuda_id)

    device = torch.device(default_cuda_id if torch.cuda.is_available() else "cpu")

    model = model_builder(setting.model_name, num_joints=21, args=setting).to(device)

    eval_function = get_EvalFunction(setting)

    print("Initialization Done, Ready to start evaluationg...\n")
    file = list_files[-1]
    model_path = os.path.join(args.path, file)
    setting = torch.load(model_path)["args"]

    model.load_state_dict(torch.load(model_path)["model"])
    model.eval()

    basepath = os.environ.get('MSRA_PATH')
    test_data_gt_file_path = os.path.join(basepath, "msra_test_groundtruth_label.txt")
    test_data_file_path = os.path.join(basepath, "msra_test_list.txt")
    gt = open(test_data_gt_file_path)
    gt.seek(0)
    gt_lines = gt.readlines()

    files = open(test_data_file_path)
    files.seek(0)
    file_lines = files.readlines()
    assert len(gt_lines) == len(file_lines)

    # socket start
    send_data = Send_data(con_socket)
    send_data.start()

    if not use_file:
        # camera open
        camera_data = Camera(con_camera)
        con_camera.acquire()
        camera_data.start()

    con_socket.acquire()

    # loop
    for i in range(len(file_lines)):
        test_set = HandDataEval(basepath=os.environ.get('MSRA_PATH'), useFile=use_file,
                                   use_default_cube=setting.use_default_cube,
                                   file_lines=file_lines[i], gt_lines=gt_lines[i], camera=None if use_file else camera_data)

        testloader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
        if not use_file:
            con_camera.notify()
            con_camera.wait()
        start_time = time.time()  # 程序开始时间
        for j, data in enumerate(testloader):
            hand_evaluate(data, test_set, eval_function)
        con_socket.notify()
        con_socket.wait()

    if not use_file:
        # camera close
        camera_data.close()
        con_camera.notify()
        con_camera.release()

    # socket close
    send_data.data_finish()
    con_socket.notify()
    con_socket.release()

