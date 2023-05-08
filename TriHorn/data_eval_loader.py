#-*-coding:utf-8-*-
from torch.utils.data.dataset import Dataset
from utils.catch_camera_image import *
from PIL import Image
import numpy as np
import os.path
import torch
import cv2
import math
import copy
import struct
from scipy import ndimage


class HandDataEval(Dataset):
    def __init__(self, basepath="", cropSize=(128, 128), numSamples = 1, useFile = True,
                 comJitter=False, RandDotPercentage=0, cropSize3D=[250, 250, 250],
                 do_norm_zero_one=False, random_seed=21,
                 drop_joint_num=0, center_refined=False,
                 use_default_cube=True, file_lines="", gt_lines="", camera=None):

        # self.fx, self.fy, self.ux, self.uy = (241., 241., 160., 120.)
        self.fx, self.fy, self.ux, self.uy = (150, 150, 160., 120.)
        self.num_joints = 21
        self.default_cubes = [250, 250, 250]
        self.useFile = useFile
        self.seqName = "test"
        if file_lines[-1] == "\n":
            file_lines = file_lines[:-1]
        self.depth_address = os.path.join(basepath, file_lines)
        self.numSamples = numSamples
        self.center_refined = center_refined
        self.do_norm_zero_one = do_norm_zero_one
        self.use_default_cube = use_default_cube
        self.cropSize = cropSize
        self.cropSize3D = cropSize3D
        self.comJitter = comJitter
        self.camera = camera

        part = gt_lines.split(' ')
        gt_uvd_original = np.zeros((self.num_joints, 3), np.float32)

        for joint in range(self.num_joints):
            for xyz in range(0, 3):
                gt_uvd_original[joint, xyz] = part[joint * 3 + xyz]

        self.data = [gt_uvd_original]

        if center_refined:
            print("Center refined being used")
            self.center_refined_uvd = [i for i in range(self.numSamples)]

        print("MSRA DatasetEval init done.")

    def __len__(self):
        return self.numSamples

    def __getitem__(self, index):

        self.valIndex = index

        if self.center_refined:
            com = self.center_refined_uvd[index]
        else:
            com = None

        data = self.LoadSample(com)

        img = self.normalizeMinusOneOne(data)

        self.sample_loaded = data

        com = torch.from_numpy(data["com3D"])

        # Image need to be HxWxC and will be divided by transform (ToTensor()), which is assumed here!
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)

        # target = torch.from_numpy(target.astype('float32'))self. = RandDotPercentage
        M_ = torch.from_numpy(data["M"])
        M_ = torch.cat([torch.cat([M_[:, :2], torch.zeros(3, 1), M_[:, 2][..., None]], dim=-1), torch.zeros(1, 4)]);
        M_[2, 2] = 1
        M_[2, 3] = 0
        M = M_.float()
        M_inv = torch.from_numpy(np.linalg.inv(data["M"]))
        M_ = torch.cat(
            [torch.cat([M_inv[:, :2], torch.zeros(3, 1), M_inv[:, 2][..., None]], dim=-1), torch.zeros(1, 4)]);
        M_[2, 2] = 1
        M_[2, 3] = 0
        M_inv = M_.float()

        cubesize = torch.from_numpy(np.array(data["cubesize"])).float()
        return img.float(), com.float(), M_inv, cubesize.float(), self.data[0]

    def cropArea3D(self, imgDepth, com, minRatioInside=0.75, size=(250, 250, 250), dsize=(128, 128)):
        """
        Crop area of hand in 3D volumina, scales inverse to the distance of hand to camera
        :param com: center of mass, in image coordinates (x,y,z), z in mm
        :param size: (x,y,z) extent of the source crop volume in mm
        :param dsize: (x,y) extent of the destination size
        :return: cropped hand image, transformation matrix for joints, CoM in image coordinates
        """
        RESIZE_BILINEAR = 0
        RESIZE_CV2_NN = 1
        RESIZE_CV2_LINEAR = 2
        CROP_BG_VALUE = 0.0
        resizeMethod = RESIZE_CV2_NN
        # calculate boundaries
        xstart, xend, ystart, yend, zstart, zend = self.comToBounds(com.copy(), size)

        # Check if part within image is large enough; otherwise stop
        xstartin = max(xstart, 0)
        xendin = min(xend, imgDepth.shape[1])
        ystartin = max(ystart, 0)
        yendin = min(yend, imgDepth.shape[0])
        ratioInside = float((xendin - xstartin) * (yendin - ystartin)) / float((xend - xstart) * (yend - ystart))
        if (ratioInside < minRatioInside) \
                and ((com[0] < 0) \
                     or (com[0] >= imgDepth.shape[1]) \
                     or (com[1] < 0) or (com[1] >= imgDepth.shape[0])):
            print("Hand largely outside image (ratio (inside) = {})".format(ratioInside))
            raise UserWarning('Hand not inside image')

        # crop patch from source
        cropped = imgDepth[max(ystart, 0):min(yend, imgDepth.shape[0]),
                  max(xstart, 0):min(xend, imgDepth.shape[1])].copy()
        # add pixels that are out of the image in order to keep aspect ratio
        cropped = np.pad(cropped, ((abs(ystart) - max(ystart, 0), abs(yend) - min(yend, imgDepth.shape[0])),
                                   (abs(xstart) - max(xstart, 0), abs(xend) - min(xend, imgDepth.shape[1]))),
                         mode='constant', constant_values=int(CROP_BG_VALUE))
        msk1 = np.bitwise_and(cropped < zstart, cropped != 0)
        msk2 = np.bitwise_and(cropped > zend, cropped != 0)
        # Backface is at 0, it is set later;
        # setting anything outside cube to same value now (was set to zstart earlier)
        cropped[msk1] = CROP_BG_VALUE
        cropped[msk2] = CROP_BG_VALUE

        wb = (xend - xstart)
        hb = (yend - ystart)
        trans = np.asmatrix(np.eye(3, dtype=float))
        trans[0, 2] = -xstart
        trans[1, 2] = -ystart
        # Compute size of image patch for isotropic scaling
        # where the larger side is the side length of the fixed size image patch (preserving aspect ratio)
        if wb > hb:
            sz = (dsize[0], int(round(hb * dsize[0] / float(wb))))
        else:
            sz = (int(round(wb * dsize[1] / float(hb))), dsize[1])

        # Compute scale factor from cropped ROI in image to fixed size image patch;
        # set up matrix with same scale in x and y (preserving aspect ratio)
        roi = cropped
        if roi.shape[0] > roi.shape[1]:  # Note, roi.shape is (y,x) and sz is (x,y)
            scale = np.asmatrix(np.eye(3, dtype=float) * sz[1] / float(roi.shape[0]))
        else:
            scale = np.asmatrix(np.eye(3, dtype=float) * sz[0] / float(roi.shape[1]))
        scale[2, 2] = 1

        # depth resize
        if resizeMethod == RESIZE_CV2_NN:
            rz = cv2.resize(cropped, sz, interpolation=cv2.INTER_NEAREST)
        else:
            raise NotImplementedError("Unknown resize method!")

        # Sanity check
        # numValidPixels = np.sum(rz != CROP_BG_VALUE)
        # if (numValidPixels < 40) or (numValidPixels < (np.prod(dsize) * 0.01)):
        #     print("Too small number of foreground/hand pixels: {}/{} ({}))".format(
        #         numValidPixels, np.prod(dsize), dsize))
        #     raise UserWarning("No valid hand. Foreground region too small.")

        # Place the resized patch (with preserved aspect ratio)
        # in the center of a fixed size patch (padded with default background values)
        ret = np.ones(dsize, np.float32) * CROP_BG_VALUE  # use background as filler
        xstart = int(math.floor(dsize[0] / 2 - rz.shape[1] / 2))
        xend = int(xstart + rz.shape[1])
        ystart = int(math.floor(dsize[1] / 2 - rz.shape[0] / 2))
        yend = int(ystart + rz.shape[0])
        ret[ystart:yend, xstart:xend] = rz
        off = np.asmatrix(np.eye(3, dtype=float))
        off[0, 2] = xstart
        off[1, 2] = ystart

        return ret, off * scale * trans, com

    def LoadSample(self, com=None):
        # Load the dataset

        if self.useFile:
            dpt = self.loadDepthMapFromFile(self.depth_address).copy()
        else:
            dpt = self.loadDepthMapFromCamera(self.camera).copy()
        original_dpt = dpt.copy()

        dpt = self.depthDealImg(dpt, minDepth=30, maxDepth=3000)
        com = self.calculateCoM(dpt, minDepth=0, maxDepth=3000)
        self.com = com

        self.randomComJitter = np.clip(np.random.randn(3) * 6, -self.comJitter,
                                       self.comJitter)  # (1 if self.comJitter else 0)* np.clip(np.random.randn(3)*6,-6,6)

        # Jitter scale (cube size)?
        cubesize = (self.default_cubes if self.use_default_cube else self.cropSize3D)

        dpt, M, com = self.cropArea3D(imgDepth=dpt, com=com, minRatioInside=0.6, size=cubesize, dsize=self.cropSize)

        com3D = self.pointImgTo3D(com)

        D = {}
        D["M"] = M
        D["com3D"] = com3D
        D["cubesize"] = cubesize
        D["dpt"] = dpt.astype(np.float32)
        D["original_dpt"] = original_dpt
        return D

    def comToBounds(self, com, size):
        """
            Calculate boundaries, project to 3D, then add offset and backproject to 2D (ux, uy are canceled)
            :param com: center of mass, in image coordinates (x,y,z), z in mm
            :param size: (x,y,z) extent of the source crop volume in mm
            :return: xstart, xend, ystart, yend, zstart, zend
            """
        zstart = com[2] - size[2] / 2.
        zend = com[2] + size[2] / 2.
        xstart = int(np.floor((com[0] * com[2] / self.fx - size[0] / 2.) / com[2] * self.fx + 0.5))
        xend = int(np.floor((com[0] * com[2] / self.fx + size[0] / 2.) / com[2] * self.fx + 0.5))
        ystart = int(np.floor((com[1] * com[2] / self.fy - size[1] / 2.) / com[2] * self.fy + 0.5))
        yend = int(np.floor((com[1] * com[2] / self.fy + size[1] / 2.) / com[2] * self.fy + 0.5))
        return xstart, xend, ystart, yend, zstart, zend

    def pointImgTo3D(self, sample):
        """
        Normalize sample to metric 3D
        :param sample: joints in (x,y,z) with x,y in image coordinates and z in mm
        :return: normalized joints in mm
        """
        ret = np.zeros((3,), np.float32)
        # convert to metric using f
        ret[0] = (sample[0] - self.ux) * sample[2] / self.fx
        ret[1] = (sample[1] - self.uy) * sample[2] / self.fy
        ret[2] = sample[2]
        return ret

    def joints3DToImg(self, sample):
        """
        Denormalize sample from metric 3D to image coordinates
        :param sample: joints in (x,y,z) with x,y and z in mm
        :return: joints in (x,y,z) with x,y in image coordinates and z in mm
        """
        ret = np.zeros((sample.shape[0], 3), np.float32)
        for i in range(sample.shape[0]):
            ret[i] = self.pointImgTo3D(sample[i])
        return ret

    def loadDepthMapFromFile(self, filename):
        """
        Read a depth-map
        :param filename: file name to load
        :return: image data of depth image
        """

        # .bin file_input
        if(filename[-3:] == "bin"):
            with open(filename, 'rb') as f:
                # first 6 uint define the full image
                width = struct.unpack('i', f.read(4))[0]
                height = struct.unpack('i', f.read(4))[0]
                left = struct.unpack('i', f.read(4))[0]
                top = struct.unpack('i', f.read(4))[0]
                right = struct.unpack('i', f.read(4))[0]
                bottom = struct.unpack('i', f.read(4))[0]
                patch = np.fromfile(f, dtype='float32', sep="")
                imgdata = np.zeros((height, width), dtype='float32')
                imgdata[top:bottom, left:right] = patch.reshape([bottom-top, right-left])
        else:
            # .jpg file_input
            im = Image.open(filename)
            imgdata = np.array(im.convert('L'), dtype='float32')
        return imgdata


    def loadDepthMapFromCamera(self, camera):
        # camera
        imgdata = camera.catch_camera_img_data()
        return imgdata

    def normalizeMinusOneOne(self, sample):
        imgD = np.asarray(sample["dpt"].copy(), 'float32')
        imgD[imgD == 0] = sample["com3D"][2] + (sample['cubesize'][2] / 2.)
        imgD -= sample["com3D"][2]
        imgD /= (sample['cubesize'][2] / 2.)
        return imgD

    def depthDealImg(self, dpt, minDepth=0, maxDepth=500):
        dpt[dpt < minDepth] = 0
        dpt[dpt > maxDepth] = 0
        return dpt

    def calculateCoM(self, dpt, minDepth=0, maxDepth=500):
        """
        Calculate the center of mass
        :param dpt: depth image
        :return: (x,y,z) center of mass
        """

        dc = dpt.copy()
        dc[dc < minDepth] = 0
        dc[dc > maxDepth] = 0
        cc = ndimage.measurements.center_of_mass(dc > 0)
        num = np.count_nonzero(dc)
        # com = np.array((cc[1] * num, cc[0] * num, dc.sum()), np.float32)

        if num == 0:
            return np.array((0, 0, 0), np.float32)
        else:
            return np.array((cc[1], cc[0], dc.sum() / num), np.float32)

    def convert_uvd_to_xyz_tensor(self, uvd):
        # uvd is a tensor of  size(B,num_joints,3)
        xyz = torch.zeros(uvd.shape)
        xyz[:, :, 2] = uvd[:, :, 2]
        xyz[:, :, 0] = (uvd[:, :, 0] - self.ux) * uvd[:, :, 2] / self.fx
        xyz[:, :, 1] = (uvd[:, :, 1] - self.uy) * uvd[:, :, 2] / self.fy
        return xyz
