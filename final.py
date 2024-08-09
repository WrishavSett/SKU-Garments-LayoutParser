# Importing libraries
import os
import shutil
import cv2
import random
import requests
import torch
import detectron2
import layoutparser as lp
from pycocotools.coco import COCO
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer


# Function to Load dataset in LayoutParser trainable format
def load_coco_annotations(annotations, coco=None):
    """
    Args:
        annotations (List):
            a list of coco annotaions for the current image
        coco (`optional`, defaults to `False`):
            COCO annotation object instance. If set, this function will
            convert the loaded annotation category ids to category names
            set in COCO.categories
    """
    layout = lp.Layout()
    for ele in annotations:
        x, y, w, h = ele['bbox']
        layout.append(
            lp.TextBlock(
                block=lp.Rectangle(x, y, w+x, h+y),
                type=ele['category_id'] if coco is None else coco.cats[ele['category_id']]['name'],
                id=ele['id']
            )
        )
    return layout


def setup_paths_and_split(coco_anno_path, coco_img_path, client_name):
    # Check if COCO dataset exists
    if not os.path.exists(coco_anno_path):
        raise FileNotFoundError(f"COCO annotations file not found at {coco_anno_path}")
    if not os.path.exists(coco_img_path):
        raise FileNotFoundError(f"COCO images path not found at {coco_img_path}")

    # Clone the git repo in the working directory(root) if not already cloned
    repo_path = './layout-model-training'
    if not os.path.exists(repo_path):
        os.system('git clone https://github.com/Layout-Parser/layout-model-training.git')

    # Copy the images folder and the result.json file to a newly made data folder inside the cloned repo
    new_data_path = os.path.join(repo_path,'data',client_name)
    if not os.path.exists(new_data_path):
        os.makedirs(new_data_path, exist_ok=True)
        shutil.copytree(coco_img_path, os.path.join(new_data_path, 'images'))
        shutil.copy(coco_anno_path, os.path.join(new_data_path, 'result.json'))

    # Splitting the dataset (result.json file) if train.json and test.json do not exist
    train_json_path = os.path.join(new_data_path, 'train.json')
    test_json_path = os.path.join(new_data_path, 'test.json')
    if not os.path.exists(train_json_path) or not os.path.exists(test_json_path):
        os.system('python ./layout-model-training/utils/cocosplit.py '
                  f'--annotation-path {os.path.join(new_data_path, "result.json")} '
                  '--split-ratio 0.80 '
                  f'--train {train_json_path} '
                  f'--test {test_json_path}')


def train_model(client_name):
    # Training LP using provided training scripts
    output_dir = f'./layout-model-training/outputs/{client_name}'
    os.system('python ./layout-model-training/tools/train_net.py '
              '--dataset_name TrainingData '
              f'--json_annotation_train ./layout-model-training/data/{client_name}/train.json '
              f'--image_path_train ./layout-model-training/data/{client_name} '
              f'--json_annotation_val ./layout-model-training/data/{client_name}/test.json '
              f'--image_path_val ./layout-model-training/data/{client_name} '
              '--config-file ./layout-model-training/configs/prima/fast_rcnn_R_50_FPN_3x.yaml '
              f'OUTPUT_DIR {output_dir} '
              'SOLVER.IMS_PER_BATCH 2 '
              )


def load_trained_model(client_name):
    # Loading the trained model for inference
    model = lp.models.Detectron2LayoutModel(
        config_path= f"./layout-model-training/outputs/{client_name}/config.yaml",
        model_path= f"./layout-model-training/outputs/{client_name}/model_final.pth",
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.4]  # Only output high accuracy preds
    )
    return model


def main():
    COCO_ANNO_PATH = './data/result.json'
    COCO_IMG_PATH = './data/images/'
    coco = COCO(COCO_ANNO_PATH)

    setup_paths_and_split(COCO_ANNO_PATH, COCO_IMG_PATH, "Client1")
    train_model("Client1")


if __name__ == "__main__":
    main()
