from bs4 import BeautifulSoup
from dataclasses import dataclass
from pathlib import Path
import argparse
import random
import cv2


@dataclass
class Mask:
    name: str
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    difficult: int
    truncated: int


@dataclass
class Annotation:
    masks: list
    segmented: int
    width: int
    height: int
    depth: int
    filename: str


def load_annotations(annotation_root_dir):
    annotations = []
    annotation_dir = Path(annotation_root_dir)
    files = [e for e in annotation_dir.iterdir() if e.is_file()]
    print(files)
    for file in files:
        path = Path(file)
        content = path.read_text()
        bs = BeautifulSoup(content, 'xml')
        objs = bs.findAll('object')
        filename = bs.find('filename').string
        width = bs.find('size').find('width').string
        height = bs.find('size').find('height').string
        depth = bs.find('size').find('depth').string
        segmented = bs.find('segmented').string

        image_annotation = Annotation([], segmented, width, height, depth, filename)

        for obj in objs:
            obj_name = obj.findChildren('name')[0].string
            difficult = int(obj.findChildren('difficult')[0].contents[0])
            truncated = int(obj.findChildren('truncated')[0].contents[0])
            bbox = obj.findChildren('bndbox')[0]
            xmin = int(bbox.findChildren('xmin')[0].contents[0])
            ymin = int(bbox.findChildren('ymin')[0].contents[0])
            xmax = int(bbox.findChildren('xmax')[0].contents[0])
            ymax = int(bbox.findChildren('ymax')[0].contents[0])
            mask = Mask(obj_name, xmin, xmax, ymin, ymax, difficult, truncated)
            image_annotation.masks.append(mask)

        annotations.append(image_annotation)

    return annotations


def process_annotation(annotation, images_root_dir):
    image_filename = str(Path(images_root_dir) / annotation.filename)
    print(image_filename)

    image = cv2.imread(image_filename)
    image = cv2.putText(image, annotation.filename, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    for mask in annotation.masks:
        box_color = (0, 255, 0)  # Green
        if mask.difficult or mask.truncated:
            box_color = (0, 0, 255)  # Red
        cv2.rectangle(image, (mask.xmin, mask.ymin), (mask.xmax, mask.ymax), box_color, 2)
        cv2.putText(image, mask.name, (mask.xmin, mask.ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-root-dir", type=str, default="../raccoons/raccoons/annotations")
    parser.add_argument("--images-root-dir", type=str, default="../raccoons/raccoons/images")

    args = vars(parser.parse_args())
    random.seed(42)

    print(f"Current directory: {Path.cwd()}")
    print(f"Home directory: {Path.home()}")

    annotations = load_annotations(args['annotation_root_dir'])
    print(annotations)
    total_images = len(annotations)
    index = random.randint(0, total_images)

    for annotation in annotations:
        image = process_annotation(annotation, args['images_root_dir'])
        cv2.imshow('image', image)
        k = chr(cv2.waitKey())
        if k == 'q':
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            break
