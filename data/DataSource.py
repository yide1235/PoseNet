import os
import torch
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np

class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True, transforms=None):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train
        self.transforms=transforms

        self.image_poses = []
        self.images_path = []

        self._get_data()

        
        


        # TODO: Define preprocessing

        normalize = T.Normalize(mean=[0.5,0.5,0.5],
                                std=[0.5,0.5,0.5])

        if not train:
            self.transforms = T.Compose(
                [T.Resize(224),
                T.RandomCrop(224),
                T.ToTensor(),
                normalize]
            )
            self.length=224

        else:
            self.transforms = T.Compose(
                [T.Resize(256),
                T.RandomCrop(224),
                T.ToTensor(),
                normalize]
            )
            self.length=224

        
        # Load mean image
        self.mean_image_path = os.path.join(self.root, 'mean_image.npy')
        if os.path.exists(self.mean_image_path):
            self.mean_image = np.load(self.mean_image_path)
            print("Mean image loaded!")
        else:
            self.mean_image = self.generate_mean_image()
        
    
        for i in range(len(self.mean_image_path[0])):
            img,_ =self.__getitem__(i)
            print(img.size())
         
            img=img-self.mean_image
   

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def generate_mean_image(self):
        print("Computing mean image:")

        # TODO: Compute mean image

        # Initialize mean_image
        
        # Iterate over all training images
        # Resize, Compute mean, etc...
        count=0
        # Store mean image
        
        mean_init=torch.zeros(self.length,self.length)
        
        for i in range(self.length):
            img,_ =self.__getitem__(i)
            mean_init=mean_init+img
            count=count+1

        mean_init=mean_init/count
        mean_image=mean_init
        print(mean_init.size())
        print("Mean image computed!")

        return mean_image

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]

        data = Image.open(img_path)

        # TODO: Perform preprocessing
        data=self.transforms(data)
        return data, img_pose

    def __len__(self):
        return len(self.images_path)