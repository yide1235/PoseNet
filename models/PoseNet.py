import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pickle
import numpy as np

def init(key, module, weights=None):
    if weights == None:
        return module

    # initialize bias.data: layer_name_1 in weights
    # initialize  weight.data: layer_name_0 in weights
    module.bias.data = torch.from_numpy(weights[(key + "_1").encode()])
    module.weight.data = torch.from_numpy(weights[(key + "_0").encode()])

    return module


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, key=None, weights=None):
        super(InceptionBlock, self).__init__()

        # TODO: Implement InceptionBlock
        # Use 'key' to load the pretrained weights for the correct InceptionBlock

        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, n1x1, kernel_size=1),
            nn.ReLU(True),
        )

        # 1x1 -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, n3x3red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.ReLU(True),
        )

        # 1x1 -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, n5x5red, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=5, padding=2),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_planes, kernel_size=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        # TODO: Feed data through branches and concatenate
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class LossHeader(nn.Module):
    def __init__(self, key, weights=None):
        super(LossHeader, self).__init__()

        # TODO: Define loss headers


    def forward(self, x):
        # TODO: Feed data through loss headers

        return xyz, wpqr


class PoseNet(nn.Module):
    def __init__(self, load_weights=True):
        super(PoseNet, self).__init__()

        # Load pretrained weights file
        if load_weights:
            print("Loading pretrained InceptionV1 weights...")
            file = open('pretrained_models/places-googlenet.pickle', "rb")
            weights = pickle.load(file, encoding="bytes")
            file.close()
        # Ignore pretrained weights
        else:
            weights = None

        # TODO: Define PoseNet layers

        self.pre_layers = nn.Sequential(
            # Example for defining a layer and initializing it with pretrained weights
            init('conv1/7x7_s2', nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3), weights),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.LocalResponseNorm(5,0.0001,0.75), 
            init('conv2/3x3_reduce', nn.Conv2d(64, 64, kernel_size=1, stride=1), weights),
            nn.ReLU(True),
            init('conv2/3x3', nn.Conv2d(64, 192, kernel_size=3, padding=1), weights),
            nn.ReLU(True),
            nn.LocalResponseNorm(5,0.0001,0.75),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Example for InceptionBlock initialization
        self._3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32, "3a", weights)
        self._3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64,"3b", weights)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self._4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64,"4a",weights)
        self._4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64,"4b",weights)
        self._4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64,"4c",weights)
        self._4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64,"4d",weights)
        self._4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128,"4e",weights)

        self._5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128,"5a",weights)
        self._5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128,"5b",weights)

        
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.avg_pool5x5 = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv1x1 = nn.Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv1x12 = nn.Conv2d(528, 128, kernel_size=1, stride=1)
        self.fc = nn.Linear(1024, 2048)
        self.fc2048 = nn.Linear(2048, 1024)

        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout7 = nn.Dropout(p=0.7)
        self.relu = nn.ReLU()
        self.cls_fc_pose_xyz = nn.Linear(2048, 3)
        self.cls_fc_pose_wpqr = nn.Linear(2048, 4)
        self.cls_fc_pose_xyz_1024 = nn.Linear(1024, 3)
        self.cls_fc_pose_wpqr_1024 = nn.Linear(1024, 4)
        print("PoseNet model created!")

    def forward(self, x):
        # TODO: Implement PoseNet forward
        out = self.pre_layers(x)

        out = self._3a(out)
        out = self._3b(out)

        out = self.max_pool(out)

        out = self._4a(out)
        cls1_pool = self.avg_pool5x5(out)
        cls1_reduction = self.conv1x1(cls1_pool)
        cls1_reduction = F.relu(cls1_reduction)
        cls1_reduction = cls1_reduction.view(cls1_reduction.size(0), -1)
        cls1_fc1 = self.fc2048(cls1_reduction)
        cls1_fc1 = self.relu(cls1_fc1)
        cls1_fc1 = self.dropout7(cls1_fc1)
        loss1_xyz = self.cls_fc_pose_xyz_1024(cls1_fc1)
        loss1_wpqr = self.cls_fc_pose_wpqr_1024(cls1_fc1)

        out = self._4b(out)
        out = self._4c(out)
        out = self._4d(out)



        cls2_pool = self.avg_pool5x5(out)
        cls2_reduction = self.conv1x12(cls2_pool)
        cls2_reduction = F.relu(cls2_reduction)
        cls2_reduction = cls2_reduction.view(cls2_reduction.size(0), -1)
        cls2_fc1 = self.fc2048(cls2_reduction)
        cls2_fc1 = self.relu(cls2_fc1)
        cls2_fc1 = self.dropout7(cls2_fc1)
        loss2_xyz = self.cls_fc_pose_xyz_1024(cls2_fc1)
        loss2_wpqr = self.cls_fc_pose_wpqr_1024(cls2_fc1)
        out = self._4e(out)

        out = self.max_pool(out)

        out = self._5a(out)
        out = self._5b(out)
        cls3_pool = self.avg_pool(out)
        cls3_pool = cls3_pool.view(cls3_pool.size(0), -1)
        cls3_fc1 = self.fc(cls3_pool)
        cls3_fc1 = self.relu(cls3_fc1)
        cls3_fc1 = self.dropout5(cls3_fc1)
        loss3_xyz = self.cls_fc_pose_xyz(cls3_fc1)
        loss3_wpqr = self.cls_fc_pose_wpqr(cls3_fc1)


        if self.training:
            return loss1_xyz, \
                   loss1_wpqr, \
                   loss2_xyz, \
                   loss2_wpqr, \
                   loss3_xyz, \
                   loss3_wpqr
        else:
            return loss3_xyz, \
                   loss3_wpqr


class PoseLoss(nn.Module):

    def __init__(self, w1_xyz, w2_xyz, w3_xyz, w1_wpqr, w2_wpqr, w3_wpqr):
        super(PoseLoss, self).__init__()

        self.w1_xyz = w1_xyz
        self.w2_xyz = w2_xyz
        self.w3_xyz = w3_xyz
        self.w1_wpqr = w1_wpqr
        self.w2_wpqr = w2_wpqr
        self.w3_wpqr = w3_wpqr


    def forward(self, p1_xyz, p1_wpqr, p2_xyz, p2_wpqr, p3_xyz, p3_wpqr, poseGT):
        # TODO: Implement loss
        # First 3 entries of poseGT are ground truth xyz, last 4 values are ground truth wpqr

        loss=nn.MSELoss()

        pose_x = poseGT[:, 0:3]
        pose_q = poseGT[:, 3:]
        
        
        # l1=torch.norm(p1_xyz-pose_x,2)+self.w1_wpqr*torch.norm(p1_wpqr-(pose_q/torch.norm(pose_q,1)),2)
        # l2=torch.norm(p2_xyz-pose_x,2)+self.w2_wpqr*torch.norm(p2_wpqr-(pose_q/torch.norm(pose_q,1)),2)
        # l3=torch.norm(p3_xyz-pose_x,2)+self.w3_wpqr*torch.norm(p3_wpqr-(pose_q/torch.norm(pose_q,1)),2)

        
        l1=loss(p1_xyz,pose_x)+self.w1_wpqr*loss(p1_wpqr,(pose_q/torch.norm(pose_q,1)))
        l2=loss(p2_xyz,pose_x)+self.w2_wpqr*loss(p2_wpqr,(pose_q/torch.norm(pose_q,1)))
        l3=loss(p3_xyz,pose_x)+self.w3_wpqr*loss(p3_wpqr,(pose_q/torch.norm(pose_q,1)))




        loss = self.w1_xyz*l1+self.w2_xyz*l2+self.w3_xyz*l3   
        return loss
