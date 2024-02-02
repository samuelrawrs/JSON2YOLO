import json
import cv2
import yaml
from collections import defaultdict
from pycocotools import mask
from utils import *


def bbox_from_keypoints(ann):
    if "keypoints" not in ann:
        return
    k = np.array(ann["keypoints"]).reshape(-1, 3)
    x_list, y_list, v_list = zip(*k)
    box = [
        min(x_list),
        min(y_list),
        max(x_list) - min(x_list),
        max(y_list) - min(y_list),
    ]
    return np.array(box, dtype=np.float64)

def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = contour[i]
        p2 = contour[i + 1] if i < num - 1 else contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1])
    return value < 0


def get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2)
            if distance_min < 0 or distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2


def merge_contours(contour1, contour2, idx1, idx2):
    contour = [contour1[i] for i in list(range(0, idx1 + 1))]
    contour.extend(contour2[i] for i in list(range(idx2, len(contour2))))
    contour.extend(contour2[i] for i in list(range(0, idx2 + 1)))
    contour.extend(contour1[i] for i in list(range(idx1, len(contour1))))
    contour = np.array(contour)
    return contour


def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)


def mask2polygon(image):
    contours, hierarchies = cv2.findContours(
        image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS
    )
    contours_approx = []
    polygons = []
    for contour in contours:
        epsilon = 0.001 * cv2.arcLength(contour, True)
        contour_approx = cv2.approxPolyDP(contour, epsilon, True)
        contours_approx.append(contour_approx)

    contours_parent = []
    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx < 0 and len(contour) >= 3:
            contours_parent.append(contour)
        else:
            contours_parent.append([])

    for i, contour in enumerate(contours_approx):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]
            if len(contour_parent) == 0:
                continue
            contours_parent[parent_idx] = merge_with_parent(contour_parent, contour)

    contours_parent_tmp = [contour for contour in contours_parent if len(contour) != 0]
    polygons = []
    for contour in contours_parent_tmp:
        polygon = contour.flatten().tolist()
        polygons.append(polygon)
    return polygons


def rle2polygon(segmentation):
    if isinstance(segmentation["counts"], list):
        segmentation = mask.frPyObjects(segmentation, *segmentation["size"])
    m = mask.decode(segmentation)
    m[m > 0] = 255
    return mask2polygon(m)


def min_index(arr1, arr2):
    """Find a pair of indexes with the shortest distance.
    Args:
        arr1: (N, 2).
        arr2: (M, 2).
    Return:
        a pair of indexes(tuple).
    """
    dis = ((arr1[:, None, :] - arr2[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(dis, axis=None), dis.shape)


def merge_multi_segment(segments):
    """Merge multi segments to one list.
    Find the coordinates with min distance between each segment,
    then connect these coordinates with one thin line to merge all
    segments into one.

    Args:
        segments(List(List)): original segmentations in coco's json file.
            like [segmentation1, segmentation2,...],
            each segmentation is a list of coordinates.
    """
    s = []
    segments = [np.array(i).reshape(-1, 2) for i in segments]
    idx_list = [[] for _ in range(len(segments))]

    # record the indexes with min distance between each segment
    for i in range(1, len(segments)):
        idx1, idx2 = min_index(segments[i - 1], segments[i])
        idx_list[i - 1].append(idx1)
        idx_list[i].append(idx2)

    # use two round to connect all the segments
    for k in range(2):
        # forward connection
        if k == 0:
            for i, idx in enumerate(idx_list):
                # middle segments have two indexes
                # reverse the index of middle segments
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segments[i] = segments[i][::-1, :]

                segments[i] = np.roll(segments[i], -idx[0], axis=0)
                segments[i] = np.concatenate([segments[i], segments[i][:1]])
                # deal with the first segment and the last one
                if i in [0, len(idx_list) - 1]:
                    s.append(segments[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segments[i][idx[0] : idx[1] + 1])

        else:
            for i in range(len(idx_list) - 1, -1, -1):
                if i not in [0, len(idx_list) - 1]:
                    idx = idx_list[i]
                    nidx = abs(idx[1] - idx[0])
                    s.append(segments[i][nidx:])
    return s


def convert_hasty_coco_json(
    json_file,
    images_dir="../coco/images/",
    output_dir="output_dir",
):
    # Check for 'train' and 'val' subdirectories in the images directory
    train_dir = Path(images_dir) / "train"
    val_dir = Path(images_dir) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise ValueError(
            "Both 'train' and 'val' directories must exist within the specified images directory."
        )

    save_dir = Path(output_dir)  # Base directory for labels
    yolo_images = save_dir / "images"

    copy_all_files(train_dir, yolo_images / "train")
    copy_all_files(val_dir, yolo_images / "val")

    coco80 = coco91_to_coco80_class()

    # Import json
    with open(json_file) as f:
        data = json.load(f)
    category_names = {
        category["id"]: category["name"] for category in data["categories"]
    }

    # Create yaml file based on category names
    yaml_path = save_dir / (save_dir.name + ".yaml")
    create_yaml_file(yaml_path, category_names)

    # Create image dict
    images = {"%g" % x["id"]: x for x in data["images"]}
    # Create image-annotations dict
    imgToAnns = defaultdict(list)
    for ann in data["annotations"]:
        imgToAnns[ann["image_id"]].append(ann)

    # Write labels file
    for img_id, anns in tqdm(imgToAnns.items(), desc=f"Annotations {json_file}"):
        img = images["%g" % img_id]
        h, w, f = img["height"], img["width"], img["file_name"]
        f = f.split("/")[-1]

        bboxes = []
        segments = []
        keypoints = []
        for ann in anns:
            # The COCO box format is [top left x, top left y, width, height]
            if len(ann["bbox"]) == 0:
                box = bbox_from_keypoints(ann)
            else:
                box = np.array(ann["bbox"], dtype=np.float64)
            box[:2] += box[2:] / 2  # xy top-left corner to center
            box[[0, 2]] /= w  # normalize x
            box[[1, 3]] /= h  # normalize y
            if box[2] <= 0 or box[3] <= 0:  # if w <= 0 and h <= 0
                continue

            cls = ann["category_id"] - 1

            box = [cls] + box.tolist()
            if box not in bboxes:
                bboxes.append(box)
            if len(ann["segmentation"]) == 0:
                segments.append([])
                continue
            if isinstance(ann["segmentation"], dict):
                ann["segmentation"] = rle2polygon(ann["segmentation"])
            if len(ann["segmentation"]) > 1:
                s = merge_multi_segment(ann["segmentation"])
                s = (
                    (np.concatenate(s, axis=0) / np.array([w, h]))
                    .reshape(-1)
                    .tolist()
                )
            else:
                s = [
                    j for i in ann["segmentation"] for j in i
                ]  # all segments concatenated
                s = (
                    (np.array(s).reshape(-1, 2) / np.array([w, h]))
                    .reshape(-1)
                    .tolist()
                )
            s = [cls] + s
            if s not in segments:
                segments.append(s)

        # Determine if the image is in 'train' or 'val' directory
        if (train_dir / f).exists():
            label_subdir = "train"
        elif (val_dir / f).exists():
            label_subdir = "val"
        else:
            print(f"Image file {f} not found in 'train' or 'val' directories.")
            continue  # Skip this image

        fn = (
            save_dir / "labels" / label_subdir / f.split(".")[0]
        )  # Adjusted path for label file
        fn.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        # Write annotations to file
        with open(fn.with_suffix(".txt"), "a") as file:
            for i in range(len(bboxes)):
                line = (
                    *(
                        segments[i]
                        if len(segments[i]) > 0
                        else bboxes[i]
                    ),
                )  # cls, box or segments
                file.write(("%g " * len(line)).rstrip() % line + "\n")


def copy_all_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)  # create output dir
    files = os.listdir(src_dir)
    for file_name in files:
        src_file = os.path.join(src_dir, file_name)
        dst_file = os.path.join(dst_dir, file_name)
        if os.path.isfile(src_file):  # check if file or directory
            shutil.copy(src_file, dst_file)


def create_yaml_file(yaml_path, category_names):
    yaml_content = {
        "path": "",  # dataset root directory in ultralytics
        "train": "images/train",  # Relative path to training images
        "val": "images/val",  # Relative path to validation images
        "test": "",  # Optional, path to test images
        "names": {
            idx: name for idx, (_, name) in enumerate(sorted(category_names.items()))
        },
    }

    with open(yaml_path, "w") as file:
        yaml.dump(yaml_content, file, sort_keys=False, default_flow_style=False)


if __name__ == '__main__':
    source = 'COCO'
    source = 'hasty-coco'
if __name__ == "__main__":
    json_file_dir = ""  # Directory containing your COCO annotations JSON file
    image_directories = ""  # Define the directories containing your training and validation images
    output_directory = ""  # Define where the output folder should be

    convert_hasty_coco_json(
        json_file=json_file_dir,
        images_dir=image_directories,
        output_dir=output_directory,
    )
    # os.system('zip -r ../coco.zip ../coco')

