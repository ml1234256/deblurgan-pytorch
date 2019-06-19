import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import ReadConcat, check_folder, image_transform, image_recovery, save_image, PSNR
import os


def test(opt, netG):
    aver_psnr = 0.0
    #aver_ssim = 0.0
    counter = 0

    test = ReadConcat(opt)
    testset = DataLoader(test, batch_size=1, shuffle=False)
    save_path = os.path.join(opt.out_dir, 'test')
    check_folder(save_path)
    netG.eval()

    for i,data in enumerate(testset):
        counter = i
        data_A = data['A']  # blur
        data_B = data['B']  # sharp
        if torch.cuda.is_available():
            data_A = data_A.cuda(opt.gpu)
            data_B = data_B.cuda(opt.gpu)
        with torch.no_grad():
            realA = Variable(data_A)
            realB = Variable(data_B)

        fakeB = netG(realA)
        # fakeB = image_recovery(fakeB.squeeze().cpu().detach().numpy())
        # realB = image_recovery(realB.squeeze().cpu().detach().numpy())
        fakeB = image_recovery(fakeB)
        realB = image_recovery(realB)

        aver_psnr += PSNR(fakeB, realB)
        # fakeB = Image.fromarray(fakeB)
        # realB = Image.fromarray(realB)
        # aver_ssim += SSIM(fakeB, realB)

        # save image
        img_path = data['img_name']
        save_image(fakeB, os.path.join(save_path, img_path[0]))
        print('save successfully {}'.format(save_path))

    aver_psnr /= counter
    # aver_ssim /= counter
    print('PSNR = %f' % (aver_psnr))


