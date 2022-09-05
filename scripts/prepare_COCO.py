import os
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import copy


DATASET = 'keycap'
data_dir = os.path.join('dataset', DATASET, 'train_seen')

CLASS_MAP = {
    "black_scratch" : 0,
    "white_scrath" : 1,
    "dent" : 2,
    "background" : 255
}
COLOR_MAP = {
    "0" : [255, 0, 0], 
    "1" : [0, 255, 0],
    "2" : [0, 0, 255]
}

def get_image_dic(path, id):
    # extract file name
    (filepath,tempfilename) = os.path.split(path)
    
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    h, w = img.shape[0], img.shape[1]
    return {
        "file_name" : tempfilename,
        "height" : h,
        "width" : w,
        "id" : id
    }

def get_bbox(mask, id):
    classes = np.unique(mask)
    ann_list = []
    cat_list = []
    for c in classes:
        img = copy.deepcopy(mask)
        img[img != c] = 255
        img[img != 255] = 0
        img = cv2.bitwise_not(img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8, ltype=None)
        contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
        rois = stats[1:]
        for roi in rois:
            ann_list.append({
                "segmenation" : [contours],
                "area" : roi[-1],
                "bbox" : roi[:-1], # (x, y, w, h)
                "category_id" : c,
                "id" : id
            })
            cat_list.append({
                "supercategory" : "seen",
                "id" : c,
                "name" : [k for k, v in CLASS_MAP.items() if v == c][0]
            })
    
    return ann_list, cat_list

    

def get_annotation_dic(path, id):
    # extract file name
    (filepath,tempfilename) = os.path.split(path)
    (filename,extension) = os.path.splitext(tempfilename)

    mask = cv2.imread(os.path.join(filepath, filename + '_label.png'), cv2.IMREAD_GRAYSCALE)
    ann_list, cate_list = get_bbox(mask, id)

    return ann_list, cate_list

def visualization(coco):
    images = coco['images']
    annotations = coco['annotations']

    for image, annotation in zip(images, annotations):
        img = cv2.imread(os.path.join(data_dir, image['file_name']), cv2.IMREAD_COLOR)
        for object in annotation:
            x, y, w, h = object['bbox']
            contours = object['segmenation']
            category_id = object['category_id']
            class_name = [k for k, v in CLASS_MAP.items() if v == category_id]
            color = COLOR_MAP[f'{category_id}']
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.drawContours(img, contours[0], -1, color, 3)
            cv2.putText(img, class_name[0], (x, y-10), 0, 1, color)
        plt.imshow(img[:, :, ::-1])
        plt.show()


def main():
    images = list()
    annotations = list()
    categories = list()
    for id, file in enumerate(glob.glob(os.path.join(data_dir, '*.bmp'))):
        images.append(get_image_dic(file, id))
        ann, cat = get_annotation_dic(file, id)
        annotations.append(ann)
        categories.append(cat)

        
    coco = {
        "images" : images,
        "annotations" : annotations,
        "categories" : categories
    }
    visualization(coco)





if __name__ == "__main__":
    main()