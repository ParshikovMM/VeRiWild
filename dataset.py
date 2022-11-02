from pathlib import Path

import albumentations as albu
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


class VeRiWildDataset():
    def __init__(
            self,
            data_path: Path,
            info_txt: str,
            transform: callable = None,
            mode: str = 'train',
    ):
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        self.vids, self.camids = set(), set()

        self.img_metas = []
        with open(self.data_path / 'train_test_split' / info_txt, "r") as f:
            for line in f.readlines():
                vid = int(line.split('/')[0])
                camid = int(line.split(' ')[2])
                self.vids.add(vid)
                self.camids.add(camid)

                self.img_metas.append({
                    'img_name': line.split(' ')[0],
                    'vid': vid,
                    'camid': camid,
                })

        self.vids, self.camids = list(self.vids), list(self.camids)

    # Возвращает общее колличество картинок
    def __len__(self):
        return len(self.img_metas)

    # Возвращает img - изображение, vid - класс машины, camid - класс камеры
    # Если mode == 'train' важно вернуть индекс vid для классификации
    # На валидации vid, camid у query и gallery должны совпадать 
    def __getitem__(self, idx):
        img_name = self.data_path / 'images' / self.img_metas[idx]['img_name']
        img = cv2.imread(str(img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']

        vid = self.img_metas[idx]['vid']
        camid = self.img_metas[idx]['camid']

        if self.mode == 'train':
            vid = self.vids.index(vid)
            camid = self.camids.index(camid)

        return img, vid, camid


def get_simple_transform(height: int = 256, width: int = 256):
    train_transform = albu.Compose([
        albu.Resize(height, width),
        albu.Normalize(),
        ToTensorV2()
    ])
    return train_transform
