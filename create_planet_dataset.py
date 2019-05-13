

"""
Created on Thr Apr 18 08:15:43 2019

@author: BATMAN/secg95
"""
import matplotlib
matplotlib.use('Agg')
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from skimage import io, transform
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pdb
from torch.utils import data
from PIL import Image


class PlanetDataset(Dataset):
    """Amazing Planet dataset."""

    def __init__(self, csv_file, root_dir, extension, problem, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.AmazonasImages = [x.split('.')[0] for x in os.listdir(root_dir)]
        self.root_dir = root_dir
        self.extension = extension
        self.labels  = pd.read_csv(csv_file)
        self.transform = transform
        self.problem = problem
        self.atmospheric = ["clear","partly_cloudy","cloudy","haze"]
        self.common = ["agriculture","bare_ground","cultivation","habitation","primary","road","water"]
        self.rare = ["artisinal_mine","blooming","blow_down","conventional_mine", "selective_logging","slash_burn"]
        self.other_tags = self.common + self.rare

    def __len__(self):

        return len(self.AmazonasImages)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir,
                                self.AmazonasImages[index]+
                                self.extension)
        image = Image.open(img_name)
        Tag =  self.labels.loc[self.labels['image_name'] == self.AmazonasImages[index]]['tags'].iloc[0]
        #labels = labels.astype('float')
        if (self.problem == 'atmospheric'):
            tags = 0
            for j in range(len(self.atmospheric)):
                if self.atmospheric[j] in Tag:
                   tags = j
                   break

            Tag = tags

        else:

            tags = np.zeros(len(self.other_tags))
            Tag = [x for x in Tag[0].split(' ') if x in self.other_tags]
            Tag = ' '.join(Tag)
            tags_train[i_img,1:] = [tag in Tag for tag in other_tags]
            Tag = torch.from_numpy(tags.astype('long'))


        if self.transform:
            image  = self.transform(image)

        sample =[image,Tag]
        return sample


