import argparse
import copy
import glob
import json
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

CLASS_MAP = {
    "black_scratch": 0,
    "white_scrath": 1,
    "dent": 2,
    "background": 255,
}
COLOR_MAP = {"0": [255, 0, 0], "1": [0, 255, 0], "2": [0, 0, 255]}


class Option:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        self.parser.add_argument(
            "--mode", type=str, default="train", help="train or test"
        )
        self.parser.add_argument(
            "--data_dir",
            type=str,
            default=os.path.join("dataset", "keycap"),
            help="dataset folder",
        )
        self.parser.add_argument(
            "--output_dir", type=str, default="fk", help="output folder"
        )
        self.parser.add_argument(
            "--debug", action="store_true", help="debug mode true or false"
        )

    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()
        args = vars(self.opt)

        self.opt.data_dir = os.path.join(self.opt.data_dir, self.opt.mode)
        self.opt.output_dir = os.path.join(
            self.opt.output_dir, self.opt.mode + "_json"
        )

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


def get_image_dic(path, id):
    # extract file name
    (filepath, tempfilename) = os.path.split(path)

    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = img.shape[0], img.shape[1]
    return {"file_name": tempfilename, "height": h, "width": w, "id": id}


def get_bbox(mask, id):
    classes = np.unique(mask)
    ann_list = []
    cat_list = []
    for c in classes:
        img = copy.deepcopy(mask)
        img[img != c] = 255
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
            # add offsets
            for contour in contours:
                contour[:, :, 0] += x1
                contour[:, :, 1] += y1

            ann_list.append(
                {
                    "segmenation": [contour.flatten().tolist() for contour in contours],
                    "area": area,
                    "bbox": bbox,  # (x, y, w, h)
                    "category_id": c,
                    "id": id,
                }
            )
            cat_list.append(
                {
                    "supercategory": "seen",
                    "id": c,
                    "name": [k for k, v in CLASS_MAP.items() if v == c][0],
                }
            )

    return ann_list, cat_list


def get_annotation_dic(path, id):
    # extract file name
    (filepath, tempfilename) = os.path.split(path)
    (filename, extension) = os.path.splitext(tempfilename)

    mask = cv2.imread(
        os.path.join(filepath, filename + "_label.png"), cv2.IMREAD_GRAYSCALE
    )
    ann_list, cate_list = get_bbox(mask, id)

    return ann_list, cate_list


def visualization(coco, opt):
    images = coco["images"]
    annotations = coco["annotations"]

    for image, annotation in zip(images, annotations):
        img = cv2.imread(
            os.path.join(opt.data_dir, image["file_name"]), cv2.IMREAD_COLOR
        )
        for object in annotation:
            x, y, w, h = object["bbox"]
            contours = [np.stack(np.split(np.array(contour), len(contour) // 2)) for contour in object["segmenation"]]
            category_id = object["category_id"]
            class_name = [k for k, v in CLASS_MAP.items() if v == category_id]
            color = COLOR_MAP[f"{category_id}"]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.drawContours(img, contours, -1, color, 3)
            cv2.putText(img, class_name[0], (x, y - 10), 0, 1, color)
        plt.imshow(img[:, :, ::-1])
        plt.show()


def main():
    opt = Option().parse()
    images = list()
    annotations = list()
    categories = list()

    ############################# make COCO #############################
    for id, file in enumerate(glob.glob(os.path.join(opt.data_dir, "*.bmp"))):
        images.append(get_image_dic(file, id))
        ann, cat = get_annotation_dic(file, id)
        annotations.append(ann)
        categories.append(cat)

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    ############################# make COCO #############################

    ############################# create json file #############################
    if not os.path.isdir(opt.output_dir):
        os.makedirs(opt.output_dir)

    string = json.dumps(coco, cls=MyEncoder)
    with open(os.path.join(opt.output_dir, opt.mode + ".json"), "w") as f:
        f.write(string)

    print("save json to ", os.path.join(opt.output_dir, opt.mode + ".json"))

    ############################# create json file #############################
    if opt.debug:
        with open(os.path.join(opt.output_dir, opt.mode + ".json")) as f:
            result = json.load(f)
        visualization(coco, opt)


if __name__ == "__main__":
    main()
