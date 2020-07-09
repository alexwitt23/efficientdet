import pathlib
import dataclasses
from typing import Tuple

import cv2
import pandas as pd
import torch


@dataclasses.dataclass
class ClfDatum:
    img_name: str
    class_idx: int


class ClfDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path: pathlib.Path, image_dir: pathlib.Path) -> None:
        self.csv_path = csv_path
        self.image_dir = image_dir

        assert self.csv_path.is_file(), f"Can't read {self.metadata_path}."
        assert self.image_dir.is_dir(), f"Can't find {self.image_dir}."

        # Load in the image data
        csv_data = pd.read_csv(self.csv_path)

        self.data = []
        for idx, row in enumerate(csv_data.iterrows()):
            class_idx = [row[1]["image_name"]] = (
                0 if row[1]["benign_malignant"] == "benign" else 1
            )
            self.data.append(ClfDatum(row[1]["image_name"], class_idx))

        self.len = len(self.data)

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        example = self.data[idx]

        img = cv2.imread(str(self.image_dir / example.img_name))
        assert img is not None, example.img_name

        return torch.Tensor(img).permute(2, 0, 1), example.class_idx