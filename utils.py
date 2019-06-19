import torch
from torch.utils.data import Dataset
from torchvision import transforms as tfs
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import math

############################ReadData/DataLoader#################################


class ReadDataset(Dataset):
    # root:Root address of image storage
    # transform: Image transform
    def __init__(self, root, transform=None):
        img_list = os.listdir(root)
        self.img_paths = [os.path.join(root, k) for k in img_list]
        self.img_name = img_list
        self.transform = transform
       
    def __getitem__(self, index):
        img_name = self.img_name[index]
        img = Image.open(self.img_paths[index]).convert('RGB')
        # img = cv.imread(self.img_paths[index], cv.IMREAD_COLOR)
        if self.transform:
            img_data = self.transform(img)
        else:
            img_data = torch.from_numpy(np.array(img))
        return {'data': img_data, 'img_name': img_name}
    
    def __len__(self):
        
        return len(self.img_paths)
    
    
class ReadConcat(Dataset):
    def __init__(self, opt):
        img_list = os.listdir(opt.dataroot)
        self.img_paths = [os.path.join(opt.dataroot, k) for k in img_list]
        self.img_name = img_list
        self.opt = opt
        transform_list = [tfs.ToTensor(),
                          tfs.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]

        self.transform = tfs.Compose(transform_list)

    def __getitem__(self, index):
        img_name = self.img_name[index]
        img = Image.open(self.img_paths[index]).convert('RGB')
        AB = img.resize((self.opt.loadSizeX * 2, self.opt.loadSizeY), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1)) # random.randint(a,b), generate an int between a and b
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
            w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
            w + w_offset:w + w_offset + self.opt.fineSize]

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)


        return {'A': A, 'B': B, 'img_name': img_name}

    def __len__(self):

        return len(self.img_paths)

##################################################################################

####################################ImageProcessing##############################


def image_transform(x):
    transform_list = []
    transform_list += [tfs.ToTensor(),
                       tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = tfs.Compose(transform_list)
    return transform(x)    


def image_recovery(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

    
    
def show_compareImage(seta, setb):
    dataset_num = len(seta)
    idxs = np.random.choice(dataset_num, 4, replace=False)
    plt.figure(figsize=(14, 6))
    for i in range(1, 5):
        plt.subplot(2, 4, i)
        plt.imshow(seta[idxs[i-1]])
    for i in range(5, 9):
        plt.subplot(2, 4, i)
        plt.imshow(setb[idxs[i-5]])
    plt.pause(0)


def save_image(image_numpy, image_path):
    if image_numpy.shape[2] == 1:
        image_numpy = np.reshape(image_numpy, (image_numpy.shape[0],image_numpy.shape[1]))
        image_pil = Image.fromarray(image_numpy, 'L')
    else:
        image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
#####################################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def update_lr(optim, lr, niter_decay):
    old_lr = lr
    lrd = lr / niter_decay
    lr -= lrd
    for param_group in optim.param_groups:
        param_group['lr'] = lr
    print('update learning rate: %f -> %f' % (old_lr, lr))
    return lr


def save_net(net, checkpoints_dir, net_name, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, net_name)
    check_folder(checkpoints_dir)
    save_path = os.path.join(checkpoints_dir, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if torch.cuda.is_available():
        net.cuda()
    print('save_net{}: {}'.format(net_name, save_filename))


def load_net(net, checkpoints_dir, net_name, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, net_name)
    save_path = os.path.join(checkpoints_dir, save_filename)
    net.load_state_dict(torch.load(save_path))
    print('load_net{}: {}'.format(net_name, save_filename))
    
    
def check_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
    
def PSNR(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    
    
    
    
    
    
    
    
    
    
    
    