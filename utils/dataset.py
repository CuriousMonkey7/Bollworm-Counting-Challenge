import os
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
from tqdm import tqdm
from multiprocessing import Pool
class FilterDataset(Dataset):
    def __init__(
        self,
        img_data,
        img_dir,
        test=False,
        transforms=None,
        preload = False
    ):
        self.img_data = img_data
        self.img_dir = img_dir
        self.transforms = transforms
        self.test = test
        self.preload =preload
        if self.preload:
            self.img = []
            with Pool() as p:
                self.img = list(tqdm(p.imap(self._read_img, range(len(img_data))),total=len(img_data),desc='Preloading: ' ))
                
                
    def _read_img(self,idx):
        img_path = os.path.join(
        self.img_dir, os.path.splitext(self.img_data.iloc[idx, 0])[0] + ".npz")
        return np.load(img_path)["img"]
    def __len__(self):
        return len(self.img_data)

    def __getitem__(self, idx):
        if not self.preload:
            img_path = os.path.join(
                self.img_dir, os.path.splitext(self.img_data.iloc[idx, 0])[0] + ".npz"
            )
            image = np.load(img_path)["img"]
        else:
            image = self.img[idx]
        # to handle PNG image, which can have more than 3 channel, we pick only first 3  channel
        image =image[:,:,:3]
    
        # if image.size(0) > 3:
        #     print(self.img_data.iloc[idx, 0])
        if self.transforms:
            image = self.transforms(image=image)["image"]
            # image = T.ColorJitter(brightness=(0, 0.57),contrast=(0, 0.44),)(image)
        if not self.test:
            label = self.img_data.iloc[idx, 1]
            return image, label
      
        return image
