import argparse
import json
import subprocess
from collections import defaultdict

import cv2
import yaml
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
    json_file, images_dir="../coco/images/", output_dir="output_dir", keep_classes=None
):
    # Check for 'train' and 'val' subdirectories in the images directory
    source_train_dir = Path(images_dir) / "train"
    source_val_dir = Path(images_dir) / "val"
    if not source_train_dir.exists() or not source_val_dir.exists():
        raise ValueError(
            "Both 'train' and 'val' directories must exist within the specified images"
            f" directory. {images_dir=}"
        )

    # copying image files to ultralytics file structure
    save_dir = Path(output_dir)  # Base directory for labels
    dest_images_dir = save_dir / "images"
    (Path(dest_images_dir) / "val").mkdir(parents=True, exist_ok=True)
    (Path(dest_images_dir) / "train").mkdir(parents=True, exist_ok=True)

    # import json
    with open(json_file) as f:
        data = json.load(f)

    # keep certain classes if keep_classes is defined
    if keep_classes:
        keep_classes = [
            x + 1 for x in keep_classes
        ]  # note: this is solely because the coco ids start from 1, while yolo ids start from 0
        keep_classes_ids = set()
        for category in data["categories"]:
            if category["id"] in keep_classes:
                keep_classes_ids.add(category["id"])

        data["categories"] = [
            cat for cat in data["categories"] if cat["id"] in keep_classes_ids
        ]

        data["annotations"] = [
            ann for ann in data["annotations"] if ann["category_id"] in keep_classes_ids
        ]

    id_mapping = {
        category["id"]: new_id for new_id, category in enumerate(data["categories"])
    }

    category_names = {
        id_mapping[category["id"]]: category["name"] for category in data["categories"]
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
        if len(anns) == 0:
            print(f"Image {f} has no annotations.")
            continue

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

            cls = id_mapping[ann["category_id"]]

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
                s = (np.concatenate(s, axis=0) / np.array([w, h])).reshape(-1).tolist()
            else:
                s = [
                    j for i in ann["segmentation"] for j in i
                ]  # all segments concatenated
                s = (np.array(s).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()
            s = [cls] + s
            if s not in segments:
                segments.append(s)

        # Determine if the image is in 'train' or 'val' directory
        if (source_train_dir / f).exists():
            label_subdir = "train"
        elif (source_val_dir / f).exists():
            label_subdir = "val"
        else:
            print(f"Image file {f} not found in 'train' or 'val' directories."
                  f" Train Path: {source_train_dir / f}, Val Path: {source_val_dir / f}")
            continue  # Skip this image

        # Copy image file to destination directory
        shutil.copy(images_dir / label_subdir / f, dest_images_dir / label_subdir / f)

        # Create the label directory and make it
        fn = (
            save_dir / "labels" / label_subdir / f.split(".")[0]
        )  # Adjusted path for label file
        fn.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists

        # Write annotations to file
        with open(fn.with_suffix(".txt"), "a") as file:
            for i in range(len(bboxes)):
                line = (
                    *(segments[i] if len(segments[i]) > 0 else bboxes[i]),
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


def create_zip_file(output_directory, remove_folder=False):
    # zip and remove the folder if indicated
    zip_file_path = Path(output_directory).with_suffix(".zip")
    result = subprocess.run(["zip", "-r", zip_file_path, output_directory], check=True)
    if remove_folder and result.returncode == 0:  # If the zip operation was successful
        shutil.rmtree(Path(output_directory))


def main():
    parser = argparse.ArgumentParser(description="Convert Hasty COCO JSON files.")
    parser.add_argument(
        "--json-file",
        "-j",
        type=Path,
        required=True,
        help="Path to the JSON Annotation file.",
    )
    parser.add_argument(
        "--images-dir",
        "-i",
        type=Path,
        required=True,
        help="Directory containing the images.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        type=Path,
        help="Output directory for the converted files.",
    )

    parser.add_argument(
        "--keep-classes",
        "-k",
        required=False,
        nargs="+",
        type=int,
        help="Enter a list of class ids to keep in the output file.",
    )

    parser.add_argument(
        "--zip",
        "-z",
        required=False,
        action="store_true",
        help="Zip the output directory if this flag is present.",
    )

    # Refer to these integers for the corresponding classes
    #   0: staple
    #   1: 'Nail: Head+Body'
    #   2: 'Nail: Head'
    #   3: 'Nail: Body'
    #   4: 'Wood: Face'
    #   5: 'Wood: Split'
    #   6: 'Metal: Other'
    #   7: 'Nail: Cut-Off'
    #   8: Hole

    args = parser.parse_args()

    convert_hasty_coco_json(
        json_file=args.json_file,
        images_dir=args.images_dir,
        output_dir=args.output_dir,
        keep_classes=args.keep_classes,
    )

    if args.zip:
        create_zip_file(output_directory=args.output_dir)

    # example usage to keep ids 0,1,2 and zip the folder:
    # python3 general_json2yolo.py -j "/home/samuel/x.json" -i "/home/samuel/images" -o /home/samuel/output -k 0 1 2 -z


if __name__ == "__main__":
    main()
