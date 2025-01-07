import pandas as pd
import numpy as np
from numpy import genfromtxt

# transform
import cv2
import albumentations as A

import libdata

class ICCAD_Data():
    def __init__(
            self, 
            # transform
            img_size: int = 384,
            interpolation: int = cv2.INTER_AREA,
        ):
        # transform
        self.img_size = img_size
        self.interpolation = interpolation

    def generate_example(self, data_idx, current, pdn_density, eff_dist, ir_drop, netlist):
        netlist_map = libdata.process(
            netlist,
            self.img_size,
            self.img_size,
        ).transpose(1, 2, 0)

        image = np.stack([
            pd.read_csv(current, sep=",", header=None).to_numpy(),
            pd.read_csv(pdn_density, sep=",", header=None).to_numpy(),
            pd.read_csv(eff_dist, sep=",", header=None).to_numpy(),
        ], axis=2)
        image = A.resize(image, self.img_size, self.img_size, interpolation=self.interpolation)
        
        ir_drop = genfromtxt(ir_drop, delimiter=",") * 1e3
        ir_drop = np.expand_dims(ir_drop, axis=0)  # 添加通道维度
            
        return {
            "data_idx": data_idx,
            "H": ir_drop.shape[1],
            "W": ir_drop.shape[2],
            "image": np.concatenate([image, netlist_map], axis=2).transpose(2, 0, 1),
            "ir_drop": ir_drop.reshape(-1, 1),
        }
