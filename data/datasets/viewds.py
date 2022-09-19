"""
@author:  davide zambrano
@contact: d.zambrano@sportradar.com

"""
from typing import Callable, Optional, Tuple
import os
import random
import dataclasses
import copy
from calib3d.calib import parameters_to_affine_transform

from calib3d.points import Point3D
import cv2


import torch
from mlworkflow import TransformedDataset, PickledDataset
import numpy as np
from PIL import Image


from deepsport_utilities.transforms import Transform
from deepsport_utilities.ds.instants_dataset.views_transforms import (
    CleverViewRandomCropperTransform,
    UndistortTransform,
)
from deepsport_utilities.transforms import IncompatibleCropException
from deepsport_utilities.utils import Subset, SubsetType

from data import transforms


class GenerateViewDS:
    """Transformed View Random Cropper Dataset"""

    def __init__(
        self,
        vds_picklefile: str = "dataset/camera_calib_viewdataset.pickle",
        output_shape: Tuple[int, int] = (1920, 1080),  # 640x360, 480x270
        num_elements: int = 1000,
        data_folder: str = "./VIEWDS",
    ) -> None:
        """
        Args:
            vds_picklefile (str, optional): _description_. Defaults to "camera_calib_viewdataset.pickle".
            output_shape (Tuple[int, int], optional): _description_. Defaults to (1920, 1080).
            num_elements (int, optional): _description_. Defaults to 1000.
        """
        absolute_path = os.path.abspath(__file__)
        absolute_path = os.path.join(*absolute_path.split("/")[:-3])

        print(f"generating data in: {absolute_path}")
        vds = PickledDataset(os.path.join("/", absolute_path, vds_picklefile))
        kwargs = {}
        kwargs["regenerate"] = True
        self.vds = TransformedDataset(
            vds,
            [
                CleverViewRandomCropperTransform(
                    output_shape=output_shape, **kwargs
                )
            ],
        )
        self.num_elements = num_elements
        self._generate_vdataset(num_elements, data_folder)

    def _generate_vdataset(self, num_elements, data_folder):
        if not os.path.isdir(data_folder):
            os.makedirs(data_folder)
        tkeys = len(self.vds.keys)
        random_keys = np.random.randint(tkeys, size=num_elements)
        for inum, random_key in enumerate(random_keys):
            fname = os.path.join(data_folder, f"{inum}")
            key = self.vds.keys[random_key]
            item = self.vds.query_item(key)
            not_generated_keys = []
            if item is not None:
                np.savez_compressed(
                    fname, image=item.image, calib=item.calib.P
                )
            else:
                not_generated_keys.append((fname, key))
        if not_generated_keys:
            print(f"not_generated_keys: {not_generated_keys}")
            self._give_it_another_try(not_generated_keys)

    def _give_it_another_try(self, not_generated_keys):
        for fname, key in not_generated_keys:
            item = self.vds.query_item(key)
            if item:
                np.savez_compressed(
                    fname, image=item.image, calib=item.calib.P
                )


class VIEWDS(torch.utils.data.Dataset):
    """A VIEW dataset that returns images and calib objects."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        download: bool = False,
        num_elements: int = 1000,
    ) -> None:
        """_summary_

        Args:
            path (_type_): _description_
        """
        if download:
            GenerateViewDS(num_elements=num_elements)
        root = "VIEWDS"
        # total = len(os.listdir(root))
        total = num_elements
        if train:
            self.list_IDs = os.listdir(root)[: int(total * 0.8)]
        else:
            self.list_IDs = os.listdir(root)[int(total * 0.8) : total]
        self.path = root
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_IDs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        fname = self.list_IDs[index]

        # Load data and get label
        item = np.load(os.path.join(self.path, f"{fname}"))
        img = Image.fromarray(item["image"])
        if self.transform is not None:
            img = self.transform(img)
        y = item["calib"].flatten()

        return img, y


class CHALLENGE(torch.utils.data.Dataset):
    """The Challenge dataset that returns transformed images."""

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        """_summary_

        Args:
            path (_type_): _description_
        """
        self.list_IDs = os.listdir(root)
        self.path = root
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.list_IDs)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        fname = f"{index}.png"

        # Load data
        oriImg = Image.open(os.path.join(self.path, f"{fname}"))
        if self.transform is not None:
            img = self.transform(oriImg)

        return img, {"img": np.array(oriImg), "index": index}


class GenerateSViewDS:
    def __init__(
        self,
        vds_picklefile: str = "dataset/camera_calib_viewdataset.pickle",
        output_shape: Tuple[int, int] = (1920, 1080),
        def_min: int = 60,
        def_max: int = 160,
        is_train: bool = False,
        train_on_full_dataset: bool = False
    ) -> None:
        """
        Args:
            vds_picklefile (str, optional): _description_. Defaults to "camera_calib_viewdataset.pickle".
            output_shape (Tuple[int, int], optional): _description_. Defaults to (1920, 1080).
            num_elements (int, optional): _description_. Defaults to 1000.
        """
        absolute_path = os.path.abspath(__file__)
        absolute_path = os.path.join(*absolute_path.split("/")[:-3])

        print(f"generating data in: {absolute_path}")
        vds = PickledDataset(os.path.join("/", absolute_path, vds_picklefile))

        transforms = [UndistortTransform()]
        if is_train:
            transforms.append(RandomFlipTransform())

        transforms.append(ApplyRandomTransform(
                    output_shape=output_shape,
                    def_min=def_min,
                    def_max=def_max,
                    regenerate=True))

        self.vds = TransformedDataset(
            vds,
            transforms,
        )
        dataset_splitter = DeepSportDatasetSplitter(
            additional_keys_usage="skip",
            train_on_full_dataset = train_on_full_dataset
        )
        (self.train, self.val, self.test) = dataset_splitter(self.vds)


def getFieldPoints():
    # 115 yd × 74 yd for soccer
    FIELD_LENGTH = 2800
    FIELD_WIDTH = 1500

    points = []
    u0 = 175
    r = 30
    u = u0
    s = 0
    for j in range(0, 7):
        for i in range(0, 13):
            points.append([i * FIELD_LENGTH / 12, FIELD_WIDTH - s, 0])
            #points.append([i * FIELD_LENGTH / 12, j * FIELD_WIDTH / 6, 0])
        s += u
        u += r

    # Add the basket center
    basket_x_shift = 120 + 15 + 45 / 2
    basket_z_pos = -305
    points.append([basket_x_shift, FIELD_WIDTH / 2, basket_z_pos])
    points.append([FIELD_LENGTH - basket_x_shift, FIELD_WIDTH / 2, basket_z_pos])

    return np.array(points, dtype=float)


class SVIEWDS(torch.utils.data.Dataset):
    """Segmentation VIEW dataset.
    It returns the segmentation target for the court.
    """

    def __init__(
        self,
        vds: GenerateSViewDS,
        transform: Optional[Callable] = None,
        return_camera: bool = False,
    ):
        "Initialization"
        self.vds = vds
        self.vds_keys = list(vds.keys)
        self.transform = transform
        self.return_camera = return_camera

        self.fieldPoints = getFieldPoints().swapaxes(0,1)

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.vds_keys)


    def getHeatmap(self, img, calib):

        oriHeight, oriWidth = img.shape[0:2]

        points = Point3D(self.fieldPoints)

        pointsOut = calib.project_3D_to_2D(points)

        pointsOut = np.array(pointsOut).swapaxes(0,1) / 4

        if hasattr(calib, "fliped"):
            for j in range(0, 7):
                pointsOut[13 * j : 13 * (j + 1)] = pointsOut[13 * j : 13 * (j + 1)][::-1]
            basket1 = pointsOut[91].copy()
            basket2 = pointsOut[92].copy()
            pointsOut[92] = basket1
            pointsOut[91] = basket2

        nbPoints = pointsOut.shape[0]

        heatmaps = np.zeros((nbPoints + 1, oriHeight // 4, oriWidth // 4), dtype=np.float32)

        # Generate the heatmaps for each key points
        for i, p in enumerate(pointsOut[:-2]):
            cv2.circle(heatmaps[i], (round(p[0]), round(p[1])), 5, (1,), -1)
            cv2.circle(heatmaps[-1], (round(p[0]), round(p[1])), 5, (1,), -1)

        # Invert last heatmaps.
        heatmaps[-1] = 1 - heatmaps[-1]

        if False:
            cv2.imshow("img", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            cv2.imshow("heatmaps", heatmaps[-1])
            cv2.waitKey()

        heatmaps = torch.tensor(heatmaps)

        return {"heatmaps" : heatmaps, "calib" : calib.P, "img" : img}


    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        key = self.vds_keys[index]
        item = self.vds.dataset.query_item(key)
        # Load data and get label
        img = Image.fromarray(item.image)

        data = self.getHeatmap(item.image, item.calib)

        if self.transform is not None:
            img = self.transform(img)

        data["index"] = index

        return (img, data)


@dataclasses.dataclass
class DeepSportDatasetSplitter:  # pylint: disable=too-few-public-methods
    validation_pc: int = 15
    additional_keys_usage: str = None
    folds: str = "ABCDE"
    split = {
        "A": ["KS-FR-CAEN", "KS-FR-LIMOGES", "KS-FR-ROANNE"],
        "B": ["KS-FR-NANTES", "KS-FR-BLOIS", "KS-FR-FOS"],
        "C": ["KS-FR-LEMANS", "KS-FR-MONACO", "KS-FR-STRASBOURG"],
        "D": ["KS-FR-GRAVELINES", "KS-FR-STCHAMOND", "KS-FR-POITIERS"],
        "E": ["KS-FR-NANCY", "KS-FR-BOURGEB", "KS-FR-VICHY"],
    }
    train_on_full_dataset: bool = False

    @staticmethod
    def count_keys_per_arena_label(keys):
        """returns a dict of (arena_label: number of keys of that arena)"""
        bins = {}
        for key in keys:
            bins[key.arena_label] = bins.get(key.arena_label, 0) + 1
        return bins

    @staticmethod
    def count_keys_per_game_id(keys):
        """returns a dict of (game_id: number of keys of that game)"""
        bins = {}
        for key in keys:
            bins[key.game_id] = bins.get(key.game_id, 0) + 1
        return bins

    def __call__(self, dataset, fold=0):
        keys = list(dataset.keys.all())
        assert 0 <= fold <= len(self.folds) - 1, "Invalid fold index"

        testing_fold = self.folds[fold]
        testing_keys = [
            k for k in keys if k.arena_label in self.split[testing_fold]
        ]

        validation_fold = self.folds[1]
        validation_keys = [
            k for k in keys if k.arena_label in self.split[validation_fold]
        ]
        remaining_folds = self.folds.replace(testing_fold, "")
        remaining_folds = remaining_folds.replace(validation_fold, "")
        remaining_arena_labels = [
            label for f in remaining_folds for label in self.split[f]
        ]
        training_keys = [
            k for k in keys if k.arena_label in remaining_arena_labels
        ]

        # Backup random seed
        random_state = random.getstate()
        random.seed(fold)

        additional_keys = [
            k
            for k in keys
            if k not in training_keys + validation_keys + testing_keys
        ]

        if additional_keys:
            if self.additional_keys_usage == "testing":
                testing_keys += additional_keys
            elif self.additional_keys_usage == "training":
                training_keys += additional_keys
            elif self.additional_keys_usage == "validation":
                validation_keys += additional_keys
            elif self.additional_keys_usage in ["none", "skip"]:
                pass
            else:
                raise ValueError(
                    "They are additional arena labels that I don't know what to do with. Please tell me the 'additional_keys_usage' argument"
                )

        # Restore random seed
        random.setstate(random_state)

        if self.train_on_full_dataset:
            training_keys += validation_keys + testing_keys

        return [
            Subset(
                name="training",
                subset_type=SubsetType.TRAIN,
                keys=training_keys,
                dataset=dataset,
            ),
            Subset(
                name="validation",
                subset_type=SubsetType.EVAL,
                keys=validation_keys,
                dataset=dataset,
            ),
            Subset(
                name="testing",
                subset_type=SubsetType.EVAL,
                keys=testing_keys,
                dataset=dataset,
            ),
        ]


class ApplyRandomTransform(CleverViewRandomCropperTransform):
    def __init__(self, *args, trials=100, def_min=60, def_max=160, **kwargs):
        """
        def -  definition in pixels per meters. (i.e. 60px/m)
        """
        super().__init__(*args, def_min=def_min, def_max=def_max, **kwargs)
        self.trials = trials

    def _apply_transform_once(self, key, item):
        if item is None:
            return None
        parameters = self._get_current_parameters(key, item)
        if parameters is None:
            return None
        keypoints, actual_size, input_shape = parameters
        try:
            angle, x_slice, y_slice = self.compute(
                input_shape, keypoints, actual_size
            )
            flip = self.do_flip and bool(np.random.randint(0, 2))
        except IncompatibleCropException:
            return None

        A = parameters_to_affine_transform(
            angle, x_slice, y_slice, self.output_shape, flip
        )
        if self.regenerate:
            item = copy.deepcopy(item)

        fliped = False
        if hasattr(item.calib, "fliped"):
            fliped = True

        ret = self._apply_transformation(item, A)
        if fliped:
            ret.calib.fliped = True
        return ret

    def __call__(self, key, item):

        for _ in range(self.trials):
            item = self._apply_transform_once(key, item)
            if not isinstance(item.image, type(None)):
                break
        return item


# Beware that calib is wrong if the image has been fliped!
# Do not use it for RandomFlipTransform evalutation
class RandomFlipTransform(Transform):

    def __call__(self, key, item):

        if random.uniform(0, 1) > 0.5:
            height, width = item.image.shape[0:2]
            F = np.float32([[-1, 0, width],
                            [ 0, 1, 0],
                            [ 0, 0, 1]])
            item.image = cv2.warpAffine(item.image, F[0:2,:], (width, height), flags=cv2.INTER_LINEAR)
            item.calib = item.calib.update(K=F@item.calib.K,
                                           width=width,
                                           height=height)
            item.calib.fliped = True # Hack to allow random flip


        return item
