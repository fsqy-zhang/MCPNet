import os
import torch
import cv2
import argparse
import numpy as np
from pprint import pprint
from tqdm import tqdm
from mmseg.apis import init_model, inference_model
"""
"""

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
# 测试图像所在文件夹
IMAGE_FILE_PATH = r""
# 模型训练结果的config配置文件路径
CONFIG = r''
# 模型训练结果的权重文件路径
CHECKPOINT = r''
# 模型推理测试结果的保存路径，每个模型的推理结果都保存在`{save_dir}/{模型config同名文件夹}`下
SAVE_DIR = r""


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('--img', default=IMAGE_FILE_PATH, help='Image file')
    parser.add_argument('--config', default=CONFIG, help='Config file')
    parser.add_argument('--checkpoint', default=CHECKPOINT, help='Checkpoint file')
    parser.add_argument('--device', default=DEVICE, help='device')
    parser.add_argument('--save_dir', default=SAVE_DIR, help='save_dir')

    args = parser.parse_args()
    return args


def make_full_path(root_list, root_path):
    file_full_path_list = []
    for filename in root_list:
        file_full_path = os.path.join(root_path, filename)
        file_full_path_list.append(file_full_path)
    return file_full_path_list


def read_filepath(root):
    from natsort import natsorted
    test_image_list = natsorted(os.listdir(root))
    test_image_full_path_list = make_full_path(test_image_list, root)
    return test_image_full_path_list

from PIL import Image
def save_colored_prediction(predictions, save_path):

#uav
    color_map = [[0, 0, 0],
                 [139, 69, 19],
                 [0, 255, 0],
                 [255, 255, 0],
                 [0, 0, 255],
                 [128, 128, 128],
                 [0, 255, 255]]

   #depglobe
    # color_map = [
    #     [0, 255, 255],   # 类别 0
    #     [255, 255, 0],   # 类别 1
    #     [255, 0, 255],   # 类别 2
    #     [0, 255, 0],     # 类别 3
    #     [0, 0, 255],     # 类别 4
    #     [255, 255, 255], # 类别 5
    #     [0, 0, 0],       # 类别 6
    # ]


    # color_map = [
    #     [0, 0, 0],   # 类别 0
    #     [255, 255, 255],   # 类别 1
    #
    # ]

    # 创建一个空的 RGB 图像
    colored_image = np.zeros((predictions.shape[0], predictions.shape[1], 3), dtype=np.uint8)

    # 将每个类别的颜色赋值到图像
    for class_id in range(len(color_map)):
        colored_image[predictions == class_id] = color_map[class_id]

    # 转换为 PIL 图像并保存
    image = Image.fromarray(colored_image)
    image.save(save_path)
def main():
    args = parse_args()

    model_mmseg = init_model(args.config, args.checkpoint, device=args.device)

    for imgs in tqdm(read_filepath(args.img)):
        result = inference_model(model_mmseg, imgs)
        pred_mask = result.pred_sem_seg.data.squeeze(0).detach().cpu().numpy().astype(np.uint8)

        save_path = os.path.join(args.save_dir, f"{os.path.basename(args.config).split('.')[0]}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saves_path=os.path.join(save_path, f"{os.path.basename(result.img_path).split('.')[0]}.png")
        save_colored_prediction(pred_mask,saves_path)

        #
        # pred_mask[pred_mask == 1] = 255
        # save_path = os.path.join(args.save_dir, f"{os.path.basename(args.config).split('.')[0]}")
        #
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        #
        # cv2.imwrite(os.path.join(save_path, f"{os.path.basename(result.img_path).split('.')[0]}.png"), pred_mask,
        #             [cv2.IMWRITE_PNG_COMPRESSION, 0])


if __name__ == '__main__':
    main()
