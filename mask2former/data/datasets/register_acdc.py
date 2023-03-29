from detectron2.data import DatasetCatalog
import os
import PIL.Image as Image
from detectron2.data import build_detection_train_loader
import torch
import copy
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper
from detectron2.data import MetadataCatalog

def acdc_train_function():
    dataset_path = r"./datasets/acdc"
    res = []
    img_path = os.path.join(dataset_path, "images", "train")
    lbl_path = os.path.join(dataset_path, "labels", "train")
    for img in os.listdir(img_path):
        file_name = os.path.join(img_path, img)
        image_id = img
        width, height = Image.open(file_name).size
        sem_seg_file_name = os.path.join(lbl_path, img)
        res.append({"file_name": file_name, "image_id": image_id, "width": width, "height": height, "sem_seg_file_name": sem_seg_file_name})

    return res

def acdc_val_function():
    dataset_path = r"./datasets/acdc"
    res = []
    img_path = os.path.join(dataset_path, "images", "val")
    lbl_path = os.path.join(dataset_path, "labels", "val")
    for img in os.listdir(img_path):
        file_name = os.path.join(img_path, img)
        image_id = img
        width, height = Image.open(file_name).size
        sem_seg_file_name = os.path.join(lbl_path, img)
        res.append({"file_name": file_name, "image_id": image_id, "width": width, "height": height, "sem_seg_file_name": sem_seg_file_name})

    return res


DatasetCatalog.register("acdc_train", acdc_train_function)
DatasetCatalog.register("acdc_val", acdc_val_function)
MetadataCatalog.get("acdc_val").gt_dir = "datasets/acdc/labels/val"
# data = DatasetCatalog.get("acdc_train")


# dataloader = build_detection_train_loader(dataset=data,
#    mapper=DatasetMapper(is_train=True, augmentations=[
#       T.Resize((800, 800))
#    ]))
#
# from detectron2.data import detection_utils as utils
#  # Show how to implement a minimal mapper, similar to the default DatasetMapper
# def mapper(dataset_dict):
#     dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#     # can use other ways to read image
#     image = utils.read_image(dataset_dict["file_name"], format="BGR")
#     # See "Data Augmentation" tutorial for details usage
#     auginput = T.AugInput(image)
#     transform = T.Resize((800, 800))(auginput)
#     image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
#     annos = [
#         utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
#         for annotation in dataset_dict.pop("annotations")
#     ]
#     return {
#        # create the format that the model expects
#        "image": image,
#        "instances": utils.annotations_to_instances(annos, image.shape[1:])
#     }
# dataloader = build_detection_train_loader(mapper=mapper)