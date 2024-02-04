# hasty.ai COCO2YOLO 

So you wanna convert your hasty.ai's COCO format to ultralytic's YOLO dataset format, you've come to the right place!
This repository is based of the ultralytics JSON2YOLO repository but is customized for exporting from hasty.ai.

## Usage

Before using this script, it's important to understand the folder structures we're converting from and into.
### File Structures

#### COCO Format (Input)
:heavy_exclamation_mark: Note that the `images` folder requires the `train` and `val` folders inside with the relevant images you exported.
```
├── <annotations-file>.json
├── images
    ├── train
        ├── <image1>.jpg
        ├── <image2>.jpg
        └── <image3>.jpg
    └── val
        ├── <image4>.jpg
        ├── <image5>.jpg
        └── <image6>.jpg
```

#### YOLO Format (Output)

```
├── <output-directory-name>
    ├── <output-directory-name>.yaml
    ├── images
        ├── train
            ├── <image1>.jpg
            ├── <image2>.jpg
            └── <image3>.jpg
        └── val
            ├── <image4>.jpg
            ├── <image5>.jpg
            └── <image6>.jpg
    ├── labels
        ├── train
            ├── <image1>.txt
            ├── <image2>.txt
            └── <image3>.txt
        └── val
            ├── <image4>.txt
            ├── <image5>.txt
            └── <image6>.txt
```

### Example Usage
Run [general_json2yolo.py](general_json2yolo.py) with the following arguments:
```
-j : (required) json directory
-i : (required) images directory (with <images>/train and <images>/val)
-o : (required) output directory
-k : (optional) ids of classes to keep based on the list below
-z : (optional) enables zipping of the folder for one less step so that you can upload it to ultralytics immediately
```
For example, running this below:
```python
python3 general_json2yolo.py -j "/home/samuel/x.json" -i "/home/samuel/images" -o /home/samuel/output -k 0 1 2 -z
```
:heavy_check_mark: chooses ids 0, 1, 2 based on the `yaml` detailing the label ids.

:heavy_check_mark: automatically zips the file for easy uploads to ultralytics.
