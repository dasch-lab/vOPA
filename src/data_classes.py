
from pathlib import Path
import pandas as pd
import os
import torch
from PIL import Image
from torchvision import transforms

# dataset for the raw vOPA images
class VOPADataset_img(torch.utils.data.Dataset):
    def __init__(
        self,
        path_img: Path = Path("src/images"),
        path_protocols: Path = Path("src/metadata/vOPA_image_metadata.csv"),
        transform=None,
    ) -> None:
        super().__init__()

        self.path_img = os.path.abspath(path_img)
        self.csv = pd.read_csv(os.path.abspath(path_protocols))
        if transform:
            self.transform = transform


    def __len__(self):
        # Returns length
        return len(self.csv)

    def __getitem__(self, idx):
        tmp_obj = self.csv.iloc[idx]
        folder_name = [_ for _ in os.listdir(self.path_img) if tmp_obj['experiment'] in _][0]        
        # convert the column into letter
        ROW_lett = ["A", "B", "C", "D", "E", "F", "G", "H"]
        ROW_num = ["01", "02", "03", "04", "05", "06", "07", "08"]
        row_lett2num = {_: x for (_, x) in zip(ROW_lett, ROW_num)}
        row = row_lett2num[tmp_obj.row]
        col = f"{tmp_obj.column:02d}"
        site = f"{tmp_obj.site:02d}"
        experiment = tmp_obj.experiment

        img_names = ["r{}c{}f{}-ch{}".format(row, col, site, ch) for ch in range(1, 5)]
        img_path = [
            # self.path_img / Path(folder_name) / Path("Images") / Path(img_name + ".png") # old version
            self.path_img / Path(folder_name) / Path(img_name + ".png")
            for img_name in img_names
        ]
        # rename the channels to be consistent cross-experiment : ch1-DAPI, ch2-CellMask, ch3-IS, ch4-EGFP
        img_path = sorted(img_path)
        img_path = self.__align_channels__(
            img_path=img_path, experiment=tmp_obj.experiment
        )
        label = img_names[0].split("-")[0]
        label = experiment + "_" + label
        xxx = torch.cat([transforms.ToTensor()(Image.open(str(_))) for _ in img_path])
        if hasattr(self, "transform"):
            xxx = self.transform(xxx)

        return xxx, label

    def __align_channels__(self, img_path, experiment):
        if experiment in ["240222_2", "240222_1"]:
            img_path[0] = Path(img_path[0].as_posix().replace("ch1", "ch4", 1))
            img_path[3] = Path(img_path[3].as_posix().replace("ch4", "ch1", 1))
        else:
            img_path = img_path

        return img_path

# dataset for the vOPA tensors
class VOPADataset(torch.utils.data.Dataset):
    def __init__(
        self,
        # path_data: Path = Path(os.path.abspath("src/tensor_dataset_pure")),
        path_data: Path = Path(os.path.abspath("src/tensor_dataset")),
        path_protocols: Path = Path(os.path.abspath("src/metadata/vOPA_tensors_metadata.csv")),
        transform=None,
        csv=None,
    ) -> None:
        super().__init__()

        self.path_data = path_data
        if csv is not None:
            self.csv = csv
        else:
            self.csv = pd.read_csv(path_protocols)

        if transform:
            self.transform = transform

    def __len__(self):
        # Returns length
        return len(self.csv)

    def __getitem__(self, idx):
        
        tmp_obj = self.csv.iloc[idx]
        experiment = tmp_obj.experiment
        
        # convert the column into letter
        ROW_lett = ["A", "B", "C", "D", "E", "F", "G", "H"]
        ROW_num = ["01", "02", "03", "04", "05", "06", "07", "08"]
        row_lett2num = {_: x for (_, x) in zip(ROW_lett, ROW_num)}
        row = row_lett2num[tmp_obj.row]
        col = f"{tmp_obj.column:02d}"
        site = f"{tmp_obj.site:02d}"
        patch = f"{tmp_obj.patch:02d}"
        tensor_name = "{}_r{}c{}f{}_{}.pt".format(experiment, row, col, site, patch)
        tensor_path = self.path_data / Path(experiment) / Path(tensor_name)
        xxx = torch.load(tensor_path)
        
        # Chech for the label col in self.csv
        if "label" in list(self.csv):
            label = tmp_obj.label
        else:
            label = "{}_r{}_c{}_f{}_{}".format(
                tmp_obj.experiment,
                row,
                col,
                site,
                patch,
            )

        if hasattr(self, "transform"):
            xxx = self.transform(xxx)

        return xxx, label
