import torch
from torch import nn
# import numpy as np

##########################################Generator##############################################


class DeblurGenerator(nn.Module):
    def __init__(self, padding_type='reflect'):
        super(DeblurGenerator, self).__init__()
        # conv-->(downsamping x 2)-->(resnblock x 9)-->(deconv x 2)-->conv-->
        deblur_model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=True),
            nn.ReLU(True)
        ]

        deblur_model += [
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=True),
            nn.ReLU(True)
        ]

        for i in range(9):
            deblur_model += [
                Resblock(256, padding_type)
            ]

        deblur_model += [
            nn.ConvTranspose2d(256, 128, 3, 2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128, track_running_stats=True),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64, track_running_stats=True),
            nn.ReLU(True),
        ]

        deblur_model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*deblur_model)

    def forward(self, x):
        res = x
        out = self.model(x)
        return torch.clamp(out + res, min=-1, max=1)


class Resblock(nn.Module):
    def __init__(self, channel, padding_type):
        super(Resblock, self).__init__()
        # conv-->instanceNorm-->relu-->conv-->instanceNorm-->
        self.conv_block = self.build_conv_block(channel, padding_type)

    def build_conv_block(self, channel, padding_type):
        conv_block = []

        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            conv_block += [nn.ZeroPad2d(1)]
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(channel, channel, kernel_size=3, padding=0),
                       nn.InstanceNorm2d(channel),
                       nn.ReLU(True)]

        conv_block += [nn.Dropout(0.5)]

        # if padding_type == 'reflect':
        #     conv_block += [nn.ReflectionPad2d(1)]
        # elif padding_type == 'replicate':
        #     conv_block += [nn.ReplicationPad2d(1)]
        # elif padding_type == 'zero':
        #     conv_block += [nn.ZeroPad2d(1)]
        # else:
        #     raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        #
        # conv_block += [nn.Conv2d(channel, channel, kernel_size=3, padding=0),
        #                nn.InstanceNorm2d(channel)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        conv_block = self.conv_block(x)
        return conv_block + x


###########################################################################################

####################################Discriminator##########################################


class DeblurDiscriminator(nn.Module):
    def __init__(self):
        super(DeblurDiscriminator, self).__init__()
        # conv-->(downsampling x 2)-->conv-->conv-->
        dis_model = [
            nn.Conv2d(3, 64, 4, 2, padding=2),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, True)
        ]

        dis_model += [
            nn.Conv2d(64, 128, 4, 2, padding=2),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, padding=2),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, True),
        ]

        dis_model += [
            nn.Conv2d(256, 512, 4, 1, padding=2),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1, 4, 1, padding=2),
            nn.Sigmoid()
        ]

        self.model = nn.Sequential(*dis_model)

    def forward(self, x):
        out = self.model(x)

        return out

################################################################################

#####################################test model#################################
# testnet = DeblurGenerator()
# test_x = Variable(torch.zeros(1,3, 72,72))
# print('################test G ####################')
# print('G_input: {}'.format(test_x.shape))
# test_y= testnet(test_x)
# print('G_output: {}'.format(test_y.shape))
#
#
# testnet = DeblurDiscriminator()
# test_x = Variable(torch.zeros(1,3, 72,72))
# print('################test D#####################')
# print('D_input: {}'.format(test_x.shape))
# test_y= testnet.forward(test_x) # 与testnet(test_x)一样
# print('D_output: {}'.format(test_y.shape))

