# import numpy as np
# from PIL import Image
# import torch
# from torch.utils.data import Dataset
#
# import os
# import json
# from collections import namedtuple
#
#
# class MapMask:
#     def __call__(self, mask):
#         mask = torch.as_tensor(np.array(mask), dtype=torch.long)
#         mapping = {0: 0, 10: 1, 20: 2, 30: 3, 40: 4, 50: 5}
#         result = torch.zeros_like(mask)
#         for old_val, new_val in mapping.items():
#             result[mask == old_val] = new_val
#         return result
#
# class Coralscapes(Dataset):
#     # Based on https://github.com/mcordts/cityscapesScripts
#     # CoralscapesClass = namedtuple('CoralscapesClass', ['name', 'id', 'train_id', 'category', 'category_id', 'ignore_in_eval', 'color'])
#     # classes = []
#     # classes.append(CoralscapesClass('unlabeled', 0, 0, 'placeholder', 0, True, (255, 255, 255)))
#     # for class_ in coralscapes_classes.keys():
#     #     classes.append(CoralscapesClass(class_, coralscapes_classes[class_], coralscapes_classes[class_], "placeholder", 0, False, coralscapes_colors[class_]))
#
#     # train_id_to_color = np.array([c.color for c in classes])
#
#
#     def __init__(self, root="../../coralscapes", split='train', transform=None, transform_target= None, refactor_flag = "", img_ext = ".png"):
#         """
#         Initialize the coralscapes.
#         Args:
#             root (str): Root directory of the coralscapes.
#             split (str, optional): The coralscapes split, one of 'train', 'test', or 'val'. Default is 'train'.
#             transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default is None.
#             transform_target (bool, optional): Whether to also transform the segmentation mask, as opposed to only the input image. Default is True.
#
#         Attributes:
#             root (str): Expanded user path of the root directory.
#             mode (str): Mode of the coralscapes, set to 'gtFine' which contains the semantic segmentation labels.
#             images_dir (str): Directory path for images.
#             targets_dir (str): Directory path for target annotations.
#             transform (callable): Transform function for images.
#             transform_target (bool): Whether to transform the mask.
#             N_classes (int): Number of classes in the coralscapes.
#             id2label (dict): Mapping of class IDs to class names.
#             label2id (dict): Mapping of class names to class IDs.
#             split (str): The coralscapes split.
#             images (list): List of image file paths.
#             targets (list): List of target file paths.
#         """
#         # global coralscapes_classes
#         # global coralscapes_colors
#
#         with open(f'{root}/{refactor_flag}classes.json', 'r') as file:
#             coralscapes_classes = json.load(file)
#             coralscapes_classes = dict(sorted(coralscapes_classes.items(), key=lambda item: item[1]))
#
#         with open(f'{root}/{refactor_flag}colors.json', 'r') as file:
#             coralscapes_colors = json.load(file)
#
#         self.CoralscapesClass = namedtuple('CoralscapesClass',
#                                                   ['name', 'id', 'train_id', 'category', 'category_id',
#                                                    'ignore_in_eval', 'color'])
#         self.classes = []
#         self.classes.append(
#             self.CoralscapesClass('unlabeled', 0, 0, 'placeholder', 0, True, (255, 255, 255)))
#
#
#         if refactor_flag :
#             for class_ in coralscapes_classes.keys():
#                 self.classes.append(
#                     self.CoralscapesClass(class_, coralscapes_classes[class_],int(coralscapes_classes[class_]/10),
#                                              "placeholder", 0, False, coralscapes_colors[class_]))
#         else :
#             for class_ in coralscapes_classes.keys():
#                 self.classes.append(
#                     self.CoralscapesClass(class_, coralscapes_classes[class_], coralscapes_classes[class_],
#                                              "placeholder", 0, False, coralscapes_colors[class_]))
#         self.flag = refactor_flag
#         self.train_id_to_color = np.array([c.color for c in self.classes])
#
#         self.root = os.path.expanduser(root)
#         self.mode = "new_mask_gt" if refactor_flag == "new_" else "gtFine"
#         self.images_dir = (
#             os.path.join(self.root, "new_img", split)
#             if refactor_flag == "new_"
#             else os.path.join(self.root, "leftImg8bit", split)
#         )
#         self.targets_dir = os.path.join(self.root, self.mode, split)
#         self.transform = transform
#         self.transform_target = transform_target
#         self.N_classes = int(1 + np.sum([dataset_class.ignore_in_eval == False for dataset_class in self.classes]))
#         self.id2label = {dataset_class.id: dataset_class.name for dataset_class in self.classes}
#         self.label2id = {v: k for k, v in self.id2label.items()}
#         self.split = split
#         self.images = []
#         self.targets = []
#
#         if split not in ['train', 'test', 'val']:
#             raise ValueError('Invalid split for mode! Please use split="train", split="test"'
#                              ' or split="val"')
#
#         if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
#             raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
#                                ' specified "split" and "mode" are inside the "root" directory')
#
#         if refactor_flag and img_ext == ".png" :
#             for file_name in os.listdir(self.images_dir):
#                 self.images.append(os.path.join(self.images_dir, file_name))
#                 target_file_name = file_name.replace(".png", "_mask.png")
#                 self.targets.append(os.path.join(self.targets_dir, target_file_name))
#         elif refactor_flag and img_ext == ".jpg" :
#             for file_name in os.listdir(self.images_dir):
#                 self.images.append(os.path.join(self.images_dir, file_name))
#                 target_file_name = file_name.replace(".jpg", "_mask.png")
#                 self.targets.append(os.path.join(self.targets_dir, target_file_name))
#         else :
#             for city in os.listdir(self.images_dir):
#                 img_dir = os.path.join(self.images_dir, city)
#                 target_dir = os.path.join(self.targets_dir, city)
#
#                 for file_name in os.listdir(img_dir):
#                     self.images.append(os.path.join(img_dir, file_name))
#                     target_file_name = file_name.replace("leftImg8bit", "gtFine")
#                     self.targets.append(os.path.join(target_dir, target_file_name))
#
#     def __getitem__(self, index):
#         """
#         Retrieve and transform an image and its corresponding segmentation map by index.
#         Args:
#             index (int): Index of the image and target to retrieve.
#         Returns:
#             tuple: A tuple containing:
#             - image (numpy.ndarray): The transformed image.
#             - target (numpy.ndarray): The transformed segmentation map.
#         """
#
#         image = np.array(Image.open(self.images[index]).convert('RGB'))
#         target = np.array(Image.open(self.targets[index]))
#
#         # if self.transform:
#         #     if self.transform_target:
#         #         transformed = self.transform(image=image, mask=target)
#         #         image = transformed["image"].transpose(2, 0, 1)
#         #         target = transformed["mask"]
#         #     else:
#         #         transformed = self.transform(image=image)
#         #         image = transformed["image"].transpose(2, 0, 1)
#         if self.transform:
#             image = self.transform(image)
#
#         if self.transform_target:
#             target = self.transform_target(target)
#             target = (target * 255).squeeze(0).long() # long tensor
#             target = target // 10
#         return image, target
#
#     def __len__(self):
#         return len(self.images)
#
#     def id_to_color_map(self, x):
#         if self.flag:
#             return self.train_id_to_color[int(x/10)]
#         else :
#             return self.train_id_to_color[x]


import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

import os
import json
from collections import namedtuple


class Coralscapes(Dataset):
    # Based on https://github.com/mcordts/cityscapesScripts
    # CoralscapesClass = namedtuple('CoralscapesClass', ['name', 'id', 'train_id', 'category', 'category_id', 'ignore_in_eval', 'color'])
    # classes = []
    # classes.append(CoralscapesClass('unlabeled', 0, 0, 'placeholder', 0, True, (255, 255, 255)))
    # for class_ in coralscapes_classes.keys():
    #     classes.append(CoralscapesClass(class_, coralscapes_classes[class_], coralscapes_classes[class_], "placeholder", 0, False, coralscapes_colors[class_]))

    # train_id_to_color = np.array([c.color for c in classes])

    def __init__(self, root="../../coralscapes", split='train', transform=None, transform_target= False, refactor_flag="",
                 img_ext=".png"):
        """
        Initialize the coralscapes.
        Args:
            root (str): Root directory of the coralscapes.
            split (str, optional): The coralscapes split, one of 'train', 'test', or 'val'. Default is 'train'.
            transform (callable, optional): A function/transform that takes in an image and returns a transformed version. Default is None.
            transform_target (bool, optional): Whether to also transform the segmentation mask, as opposed to only the input image. Default is True.

        Attributes:
            root (str): Expanded user path of the root directory.
            mode (str): Mode of the coralscapes, set to 'gtFine' which contains the semantic segmentation labels.
            images_dir (str): Directory path for images.
            targets_dir (str): Directory path for target annotations.
            transform (callable): Transform function for images.
            transform_target (bool): Whether to transform the mask.
            N_classes (int): Number of classes in the coralscapes.
            id2label (dict): Mapping of class IDs to class names.
            label2id (dict): Mapping of class names to class IDs.
            split (str): The coralscapes split.
            images (list): List of image file paths.
            targets (list): List of target file paths.
        """
        # global coralscapes_classes
        # global coralscapes_colors

        with open(f'{root}/{refactor_flag}classes.json', 'r') as file:
            coralscapes_classes = json.load(file)
            coralscapes_classes = dict(sorted(coralscapes_classes.items(), key=lambda item: item[1]))

        with open(f'{root}/{refactor_flag}colors.json', 'r') as file:
            coralscapes_colors = json.load(file)

        self.CoralscapesClass = namedtuple('CoralscapesClass',
                                           ['name', 'id', 'train_id', 'category', 'category_id',
                                            'ignore_in_eval', 'color'])
        self.classes = []
        self.classes.append(
            self.CoralscapesClass('unlabeled', 0, 0, 'placeholder', 0, True, (255, 255, 255)))

        if refactor_flag:
            for class_ in coralscapes_classes.keys():
                self.classes.append(
                    self.CoralscapesClass(class_, coralscapes_classes[class_], int(coralscapes_classes[class_] / 10),
                                          "placeholder", 0, False, coralscapes_colors[class_]))
        else:
            for class_ in coralscapes_classes.keys():
                self.classes.append(
                    self.CoralscapesClass(class_, coralscapes_classes[class_], coralscapes_classes[class_],
                                          "placeholder", 0, False, coralscapes_colors[class_]))
        self.flag = refactor_flag
        self.train_id_to_color = np.array([c.color for c in self.classes])

        self.root = os.path.expanduser(root)
        self.mode = "new_mask_gt" if refactor_flag == "new_" else "gtFine"
        self.images_dir = (
            os.path.join(self.root, "new_img", split)
            if refactor_flag == "new_"
            else os.path.join(self.root, "leftImg8bit", split)
        )
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.transform = transform
        self.transform_target = transform_target
        self.N_classes = int(1 + np.sum([dataset_class.ignore_in_eval == False for dataset_class in self.classes]))
        self.id2label = {dataset_class.id: dataset_class.name for dataset_class in self.classes}
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.split = split
        self.images = []
        self.targets = []

        if split not in ['train', 'test', 'val']:
            raise ValueError('Invalid split for mode! Please use split="train", split="test"'
                             ' or split="val"')

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.targets_dir):
            raise RuntimeError('Dataset not found or incomplete. Please make sure all required folders for the'
                               ' specified "split" and "mode" are inside the "root" directory')

        if refactor_flag and img_ext == ".png":
            for file_name in os.listdir(self.images_dir):
                self.images.append(os.path.join(self.images_dir, file_name))
                target_file_name = file_name.replace(".png", "_mask.png")
                self.targets.append(os.path.join(self.targets_dir, target_file_name))
        elif refactor_flag and img_ext == ".jpg":
            for file_name in os.listdir(self.images_dir):
                self.images.append(os.path.join(self.images_dir, file_name))
                target_file_name = file_name.replace(".jpg", "_mask.png")
                self.targets.append(os.path.join(self.targets_dir, target_file_name))
        else:
            for city in os.listdir(self.images_dir):
                img_dir = os.path.join(self.images_dir, city)
                target_dir = os.path.join(self.targets_dir, city)

                for file_name in os.listdir(img_dir):
                    self.images.append(os.path.join(img_dir, file_name))
                    target_file_name = file_name.replace("leftImg8bit", "gtFine")
                    self.targets.append(os.path.join(target_dir, target_file_name))

    def __getitem__(self, index):
        """
        Retrieve and transform an image and its corresponding segmentation map by index.
        Args:
            index (int): Index of the image and target to retrieve.
        Returns:
            tuple: A tuple containing:
            - image (numpy.ndarray): The transformed image.
            - target (numpy.ndarray): The transformed segmentation map.
        """

        image = np.array(Image.open(self.images[index]).convert('RGB'))
        target = np.array(Image.open(self.targets[index]))

        if self.transform:
            transformed = self.transform(image=image, mask=target)
            image = transformed['image']
            target = transformed['mask']
            target = torch.as_tensor(target,dtype=torch.long)
            target = target // 10
        # # Xử lý mask
        # target = torch.as_tensor(target, dtype=torch.long)
        # if self.flag:
        #     target = target // 10

        return image, target

    def __len__(self):
        return len(self.images)

    def id_to_color_map(self, x):
        if self.flag:
            return self.train_id_to_color[int(x / 10)]
        else:
            return self.train_id_to_color[x]










