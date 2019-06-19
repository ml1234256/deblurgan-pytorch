import argparse
from model import DeblurGenerator, DeblurDiscriminator
from train import train
from test import test
from utils import weights_init, load_net
import torch

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='train', help='train or test')
parser.add_argument('--dataroot', default='./data/dataset/concat_AB', help='path to dataset')
parser.add_argument('--out_dir', default='./result', help='output direction')
parser.add_argument('--loadSizeX', type=int, default=360, help='scale images to this size')
parser.add_argument('--loadSizeY', type=int, default=360, help='scale images to this size')
parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
parser.add_argument('--epoch', type=int, default=1, help='the starting epoch count')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--niter', type=int, default=150, help='of iter at starting learning rate')
parser.add_argument('--niter_decay', type=int, default=150, help='of iter to linearly decay learning rate to zero')
parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end '
                                                                   'of epochs')
parser.add_argument('--checkpoints_dir', default='./checkpoints', help='The direction model saved')
parser.add_argument('--load_epoch', type=int, default=1, help='load epoch checkpoint')
parser.add_argument('--gpu', default=0, help='gpu_id')
parser.add_argument('--no_flip', default=True, help='if specified, do not flip the images for data augmentation')

opt = parser.parse_args()

torch.backends.cudnn.benchmark = True

if opt.model == 'train':
    netG = DeblurGenerator().apply(weights_init)
    netD = DeblurDiscriminator().apply(weights_init)
    if torch.cuda.is_available():
        netG = netG.cuda(opt.gpu)
        netD = netD.cuda(opt.gpu)
    optim_G = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optim_D = torch.optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    train(opt, netG, netD, optim_G, optim_D)

if opt.model == 'test':
    netG = DeblurGenerator()
    load_net(netG, opt.checkpoints_dir, 'G', opt.load_epoch)
    if torch.cuda.is_available():
        netG = netG.cuda(opt.gpu)

    test(opt, netG)
# import matplotlib.pyplot as plt
# a = ReadConcat(opt, transform=image_transform)
# img = a[10]['A']
# print(type(img))
# print(img.shape)
# #img = image_recovery(img)
# img = img.cpu().float().numpy()
# img = (np.transpose(img, (1, 2, 0)) + 1) / 2.0 * 255.0
# img = img.astype(np.uint8)
# plt.imshow(img)
# plt.pause(0)
# # print(img.shape)


# plt.imshow(image_recovery(img))
# plt.pause(0)