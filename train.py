import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
from models.PoseNet import PoseNet, PoseLoss
from data.DataSource import *
import os
from optparse import OptionParser
import matplotlib.pyplot as plt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main(epochs, batch_size, learning_rate, save_freq, data_dir):
    # train dataset and train loader
    
    datasource = DataSource(data_dir, train=True)

    train_size=0.8*len(datasource)
    train_size=int(train_size)
    train_dataset=[]
    test_dataset=[]


    train_loss=[]
    test_loss=[]

    for i in range(train_size):
        train_dataset.append(datasource[i])
    for i in range(train_size,len(datasource)):
        test_dataset.append(datasource[i])


    #train_loader = Data.DataLoader(dataset=datasource, batch_size=batch_size, shuffle=True)
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader=Data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


    

    # load model
    posenet = PoseNet().to(device)

    # loss function
    criterion = PoseLoss(0.3, 0.3, 1., 300, 300, 300)

    # train the network
    optimizer = torch.optim.Adam(nn.ParameterList(posenet.parameters()),
                     lr=learning_rate, eps=1,
                     weight_decay=0.0625,
                     betas=(0.9, 0.999))

    batches_per_epoch = len(train_loader.batch_sampler)

    batches_per_epoch2 = len(test_loader.batch_sampler)    

    for epoch in range(epochs):
        print("Starting epoch {}:".format(epoch))
        posenet.train()
        trainloss_ave=0
        testloss_ave=0
        for step, (images, poses) in enumerate(train_loader):
            b_images = Variable(images, requires_grad=True).to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
        #torch.save(posenet.state_dict(), save_path)
        #print("Network saved!")

            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)
            b_poses = Variable(torch.Tensor(poses), requires_grad=True).to(device)

            p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
            loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("{}/{}: loss = {}".format(step+1, batches_per_epoch, loss))
  
            trainloss_ave+=float(format(loss))
        
        trainloss_ave=trainloss_ave/len(train_loader)
    
        print("average train loss for this epoch is"+str(trainloss_ave))
        train_loss.append(trainloss_ave)





        print("start test for this epoch:")
        print('/n')

        for step, (images, poses) in enumerate(test_loader):
            b_images = Variable(images, requires_grad=True).to(device)
            poses[0] = np.array(poses[0])
            poses[1] = np.array(poses[1])
        #torch.save(posenet.state_dict(), save_path)
        #print("Network saved!")

            poses[2] = np.array(poses[2])
            poses[3] = np.array(poses[3])
            poses[4] = np.array(poses[4])
            poses[5] = np.array(poses[5])
            poses[6] = np.array(poses[6])
            poses = np.transpose(poses)
            b_poses = Variable(torch.Tensor(poses), requires_grad=True).to(device)

            p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
            loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("{}/{}: loss = {}".format(step+1, batches_per_epoch, loss))
            testloss_ave+=float(format(loss))
        
        testloss_ave=testloss_ave/len(test_loader)
        print("average test loss for this epoch is"+str(testloss_ave))
        test_loss.append(testloss_ave)






        # Save state
        if epoch % save_freq == 0:
            save_filename = 'epoch_{}.pth'.format(str(epoch+1).zfill(5))
            save_path = os.path.join('checkpoints', save_filename)
            torch.save(posenet.state_dict(), save_path)


    train_loss_np=np.array(train_loss)
    test_loss_np=np.array(test_loss)

    plt.plot(train_loss_np)
    plt.plot(test_loss_np)
    plt.show()


def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', default=400, type='int')
    parser.add_option('--learning_rate', default=0.00007)
    parser.add_option('--batch_size', default=35, type='int')
    parser.add_option('--save_freq', default=20, type='int')
    parser.add_option('--data_dir', default='/data/datasets/KingsCollege')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    main(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate, save_freq=args.save_freq, data_dir=args.data_dir)
