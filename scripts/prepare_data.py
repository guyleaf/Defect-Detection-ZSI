import argparse
import copy
import glob
import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt


SEEN_CLASSES = ["black_scratch", "white_scrath", "dent"]
COLOR_MAP = {"1": [255, 0, 0], "2": [0, 255, 0], "3": [0, 0, 255]}


class Option:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        self.parser.add_argument(
            "--mode", type=str, default="train_seen", help="train or test"
        )
        self.parser.add_argument(
            "--data_dir",
            type=str,
            default=os.path.join("dataset", "keycap"),
            help="dataset folder",
        )
        self.parser.add_argument(
            "--output_dir",
            type=str,
            default="annotations",
            help="output folder",
        )
        self.parser.add_argument(
            "--debug", action="store_true", help="debug mode true or false"
        )

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        self.opt.data_dir = os.path.join(self.opt.data_dir, self.opt.mode)

        print("------------ Options -------------")
        for k, v in sorted(args.items()):
            print("%s: %s" % (str(k), str(v)))
        print("-------------- End ----------------")
        return self.opt


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def get_category_dic(
    category_list: list[str], supercategory: str = "seen"
):
    categories = []
    for idx, name in enumerate(category_list):
        category = {"id": idx + 1, "name": name, "supercategory": supercategory}
        categories.append(category)
    return categories


def get_image_dic(path, id):
    # extract file name
    _, tempfilename = os.path.split(path)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = img.shape[0], img.shape[1]
    return {"file_name": tempfilename, "height": h, "width": w, "id": id}


def get_bbox(mask, image_id, start_ann_id):
    gray_scales = np.unique(mask)
    ann_list = []
    ann_id = start_ann_id

    for gray_scale in gray_scales:
        if gray_scale == 255:
            continue

        img = copy.deepcopy(mask)
        img[img != gray_scale] = 255
        img[img != 255] = 0
        img = cv2.bitwise_not(img)
        (
            num_labels,
            labels,
            stats,
            centroids,
        ) = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=None)

        rois = stats[1:]
        for roi in rois:
            area = roi[-1]
            bbox = roi[:-1]
            x1, y1, x2, y2 = (
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3],
            )

            contours, _ = cv2.findContours(
                img[y1:y2, x1:x2],
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            # add x, y offsets
            for contour in contours:
                contour[:, :, 0] += x1
                contour[:, :, 1] += y1

            if len(contours) == 0:
                print("segmentation is empty. discarded")
                continue

            ann_list.append(
                {
                    "iscrowd": 0,
                    "segmentation": [contour.flatten() for contour in contours],
                    "area": area,
                    "bbox": bbox,  # (x, y, w, h)
                    "category_id": gray_scale + 1,
                    "image_id": image_id,
                    "id": ann_id,
                }
            )
            ann_id += 1

    return ann_list


def get_annotation_dic(path, image_id, start_ann_id=0):
    # extract file name
    filepath, tempfilename = os.path.split(path)
    filename, _ = os.path.splitext(tempfilename)

    mask = cv2.imread(
        os.path.join(filepath, filename + "_label.png"), cv2.IMREAD_GRAYSCALE
    )

    ann_list = get_bbox(mask, image_id, start_ann_id)
    return ann_list


def visualization(coco, opt):
    images = coco["images"]
    annotations = coco["annotations"]

    for image, annotation in zip(images, annotations):
        file_name = image["file_name"]
        img = cv2.imread(
            os.path.join(opt.data_dir, image["file_name"]), cv2.IMREAD_COLOR
        )
        contoured_img = copy.deepcopy(img)
        for object in annotation:
            x, y, w, h = object["bbox"]
            contours = [
                np.stack(np.split(contour, len(contour) // 2))
                for contour in object["segmentation"]
            ]
            category_id = object["category_id"]
            class_name = SEEN_CLASSES[category_id - 1]
            color = COLOR_MAP[f"{category_id}"]

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(contoured_img, (x, y), (x + w, y + h), color, 2)
            cv2.drawContours(contoured_img, contours, -1, color, 3)
            cv2.putText(img, class_name, (x, y - 10), 0, 1, color)
            cv2.putText(contoured_img, class_name, (x, y - 10), 0, 1, color)
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.title(file_name)
        plt.imshow(img[:, :, ::-1])
        plt.subplot(1, 2, 2)
        plt.title(file_name)
        plt.imshow(contoured_img[:, :, ::-1])
        plt.show()


def main():
    opt = Option().parse()
    images = []
    annotations = []
    categories = get_category_dic(SEEN_CLASSES, "seen")

    ############################# make COCO #############################
    start_ann_id = 0
    for id, file in enumerate(glob.glob(os.path.join(opt.data_dir, "*.bmp"))):
        images.append(get_image_dic(file, id))
        annotations.extend(get_annotation_dic(file, id, start_ann_id))
        start_ann_id += len(annotations[-1])

    coco = {
        "info": {},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    ############################# make COCO #############################

    ############################# create json file #############################
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    with open(os.path.join(opt.output_dir, opt.mode + ".json"), "w") as f:
        json.dump(coco, f, sort_keys=True, cls=MyEncoder)

    print("save json to ", os.path.join(opt.output_dir, opt.mode + ".json"))

    ############################# create json file #############################
    if opt.debug:
        visualization(coco, opt)


if __name__ == "__main__":
    main()
