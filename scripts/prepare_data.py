import glob
import os
from pathlib import Path
from typing import Iterable
import cv2
from tap import Tap


MASK_TYPE = "png"
TRAIN_CATEGORY_IDS = {"scratch": 1}
TEST_CATEGORY_IDS = {"scratch": 1}


def create_category_annotation(category_ids):
    category_list = []
    for key, value in category_ids.items():
        category = {"id": value, "name": key, "supercategory": key}
        category_list.append(category)
    return category_list


def create_image_annotation(image_id, file_name, width, height):
    return {
        "id": image_id,
        "width": width,
        "height": height,
        "file_name": file_name,
    }


def create_annotation_format(
    bbox, area, segmentation, image_id, category_id, annotation_id
):
    return {
        "iscrowd": 0,
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "segmentation": [segmentation],
    }


def get_coco_json_format():
    return {
        "info": {},
        "licenses": [],
        "images": [],
        "categories": [],
        "annotations": [],
    }


def find_contours(image: cv2.Mat):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(
        gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def get_image_annotations(path: str):
    image = cv2.imread(path)
    contours = find_contours(image)
    cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
    annotations = []
    for contour in contours:
        # print(contour)
        rect = cv2.boundingRect(contour)
        x, y, w, h = rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(
            image,
            "Scratch Detected",
            (x, y - 10),
            0,
            0.5,
            (0, 0, 0),
        )
        annotations.append(
            {
                "bbox": cv2.boundingRect(contour),
                "area": cv2.contourArea(contour),
                "segmentation": contour.flatten().tolist(),
            }
        )
    cv2.imshow("contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return annotations


class ArgumentParser(Tap):
    data_dir: Path  # the root of data folder
    output_dir: Path = Path(os.getcwd()) / "annotations"


def main():
    args = ArgumentParser().parse_args()
    coco_format = get_coco_json_format()

    for mode in ["train"]:
        image_dir = str(args.data_dir / mode / "images" / f"*.{MASK_TYPE}")
        for image_name in glob.glob(image_dir):
            annotations = get_image_annotations(image_name)
            # print(annotations)


if __name__ == "__main__":
    main()
