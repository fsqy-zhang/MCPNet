# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

import mmengine.fileio as fileio

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DeepGlobeDataset(BaseSegDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    METAINFO = dict(
        classes=('urban', 'agriculture', 'rangeland',  'forest',
                 'water', 'barren','unkown'),
        PALETTE=[[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0],
                 [0, 0, 255], [255, 255, 255],[0,0,0]]
    )

    def __init__(self,
                 img_suffix='_sat.jpg',
                 seg_map_suffix='_mask.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs)

