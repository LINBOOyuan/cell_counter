import argparse
import os
import numpy as np
import time

from modeling.deeplab import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import *
from torchvision.utils import make_grid, save_image
import cv2
'''
# python demo.py --in-path your_file --out-path ypur_dst_file
# python demo.py --in-path /home/cwlab/deeplabv3P/pytorch-deeplab-xception/Dish_aug_dino/testing_original --out-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/test_result --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/Dish_aug_dino/deeplabDishXception/model_best.pth.tar
# python demo.py --in-path /home/cwlab/deeplabv3P/pytorch-deeplab-xception/Dish_aug_dino/testing_original --out-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/test_result_Resnet --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/Dish_aug_dino/deeplabDishResnet/model_best.pth.tar
# python demo.py --in-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/Dish_dino_low/20220118/Patch_data_testing/Original --out-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/Low_dish_0119_mobilenet --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/Dish_dino_low/deeplabDish_low_Mobilenet/model_best.pth.tar
# python demo.py --in-path /media/windows/database/trash/0124/img --out-path /media/windows/database/trash/0124/Low_dish_xception --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/Dish_dino_low/deeplabDish_low_Xception/model_best.pth.tar
# python demo.py --in-path /media/windows/database/trash/Testing_original_BC --out-path /media/windows/database/trash/Testing_original_BC_result/mobilenet --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/BreastCancer_dino/Dino_BC_Mobilenet/model_best.pth.tar
# python demo.py --in-path /media/windows/database/trash/Testing_original_BC --out-path /media/windows/database/trash/Testing_original_BC_result/resnet --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/BreastCancer_dino/Dino_BC_Resnet/model_best.pth.tar
# python demo.py --in-path /media/windows/database/trash/Testing_original_BC --out-path /media/windows/database/trash/Testing_original_BC_result/xception --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/BreastCancer_dino/Dino_BC_Xception/model_best.pth.tar

# python demo.py --in-path /media/cwlab/KINGSTON/New_dataset/Testing_original/IT-16 --out-path /media/cwlab/KINGSTON/New_dataset/testing_mask_resnet/IT-16 --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/BreastCancer_dino/Dino_BC_Resnet/model_best.pth.tar
# python demo.py --in-path /media/cwlab/KINGSTON/New_dataset/Testing_original/IT-34 --out-path /media/cwlab/KINGSTON/New_dataset/testing_mask_xception/IT-34 --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/BreastCancer_dino/Dino_BC_Xception/model_best.pth.tar
# python demo.py --in-path /media/cwlab/KINGSTON/New_dataset/Testing_original/J38 --out-path /media/cwlab/KINGSTON/New_dataset/testing_mask_xception/J38 --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/BreastCancer_dino/Dino_BC_Xception/model_best.pth.tar

# python demo.py --in-path /media/cwlab/KINGSTON/New_dataset/Testing_original/J38 --out-path /media/cwlab/KINGSTON/New_dataset/testing_mask_resnet/J38 --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/BreastCancer_dino/Dino_BC_Resnet/model_best.pth.tar
# python demo.py --in-path /media/windows/database/trash/0223_BC_Best --out-path /media/windows/database/trash/2223_BC_Best_result/xception --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/BreastCancer_dino/Dino_BC_Xception/model_best.pth.tar

# python demo.py --in-path /media/cwlab/KINGSTON/New_dataset/Testing_original/IT-16 --out-path /media/cwlab/KINGSTON/New_dataset/0303_BC/xception --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/BreastCancer_dino/Dino_BC_Xception/model_best.pth.tar
# python demo.py --in-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/fish_deeplab_0315/FISH/Fish_200_update_test/ori_img --out-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/fish_deeplab_0315/FISH/Fish_200_update_test/result/xception --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/Fish_leo/Fish_leo/model_best.pth.tar
# python demo.py --in-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/fish_deeplab_0315/FISH/Fish_200_update_test/ori_img --out-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/fish_deeplab_0315/FISH/Fish_200_update_test/result/resnet --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/Fish_leo/Fish_leo_resnet/model_best.pth.tar
# python demo.py --in-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/fish_deeplab_0315/FISH/Fish_200_update_test/ori_img --out-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/fish_deeplab_0315/FISH/Fish_200_update_test/result/mobilenet --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/Fish_leo/Fish_leo_mobilenet/model_best.pth.tar
# python demo.py --in-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/fish_deeplab_0315/FISH/Fish_200_update_test/ori_img --out-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/fish_deeplab_0315/FISH/Fish_200_update_test/result/coatnet --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/Fish_leo/Fish_leo_test_develop/experiment_10/checkpoint.pth.tar

# python demo.py --in-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/cell_count/ori --out-path /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/cell_count/result/resnet --ckpt /media/cwlab/68b50911-f826-4575-8b56-6c1f7a09ef45/deeplabv3P/pytorch-deeplab-xception/run/cell_count/cell_count_Resnet/model_best.pth.tar
'''
class deeplabv3P_demo:

    def __init__(self,in_path,out_path,ckpt_path):
        # self.in_path = in_path
        # self.out_path = out_path
        # self.ckpt_path = ckpt_path
        self.parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
        self.parser.add_argument('--in-path', type=str,default=in_path, help = 'image to test')
        self.parser.add_argument('--out-path', type=str,default=out_path, help='mask image to save')
        self.parser.add_argument('--backbone', type=str, default='resnet',
                            choices=['resnet', 'xception', 'drn', 'mobilenet','coatnet'],
                            help='backbone name (default: resnet)')
        self.parser.add_argument('--ckpt', type=str, default=ckpt_path,
                            help='saved model')                        
        self.parser.add_argument('--out-stride', type=int, default=16,
                            help='network output stride (default: 8)')
        self.parser.add_argument('--no-cuda', action='store_true', default=
                            False, help='disables CUDA training')
        self.parser.add_argument('--gpu-ids', type=str, default='0',
                            help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')
        self.parser.add_argument('--dataset', type=str, default='cell_count', #modify to dish
                            choices=['pascal', 'coco', 'cityscapes','Dish_aug_dino','Dish_dino_low','BreastCancer_dino','Fish_leo','cell_count'], #0104 dino
                            help='dataset name (default: pascal)')
        self.parser.add_argument('--crop-size', type=int, default=513,
                            help='crop image size')
        self.parser.add_argument('--num_classes', type=int, default=4, help = 'crop image size')
        self.parser.add_argument('--sync-bn', type = bool, default=None, help='whether to use sync bn (default : auto')
        self.parser.add_argument('--freeze-bn', type=bool, default=False,
                            help='whether to freeze bn parameters (default: False)')


    def get_demo_result(self):
        args = self.parser.parse_args()
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        if args.cuda:
            try:
                args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
            except ValueError:
                raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

        if args.sync_bn is None:
            if args.cuda and len(args.gpu_ids) > 1:
                args.sync_bn = True
            else:
                args.sync_bn = False

        model_s_time = time.time()
        model = DeepLab(num_classes = args.num_classes,
                        backbone = args.backbone,
                        output_stride = args.out_stride,
                        sync_bn=args.out_stride,
                        freeze_bn = args.freeze_bn
                        )

        # ckpt = torch.load(args.ckpt, map_location = 'cpu')
        ckpt = torch.load(args.ckpt, map_location = 'cpu')
        model.load_state_dict(ckpt['state_dict'])
        model= model.cuda()
        model_u_time = time.time()
        model_load_time = model_u_time-model_s_time
        print("model load time is {}".format(model_load_time))

        composed_transfroms = transforms.Compose([tr.Normalize(mean=(0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)),
            tr.ToTensor()])
        for name in os.listdir(args.in_path):
            s_time = time.time()
            image = Image.open(args.in_path+"/"+name).convert('RGB')

            #image = Image.open(self.in_path).convert('RGB')
            target = Image.open(args.in_path+"/"+name).convert('L')
            sample = {'image': image, 'label': target}
            tensor_in = composed_transfroms(sample)['image'].unsqueeze(0)

            model.eval()
            if args.cuda:
                tensor_in = tensor_in.cuda()
            with torch.no_grad():
                output = model(tensor_in)
            
            grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3],1)[1].detach().cpu().numpy()),
                                    3, normalize = False, range=(0,255))
            # save_image (grid_image,self.in_path+"/"+"{}_mask.bmp".format(name[0:-4]))
            save_image(grid_image, args.out_path+"/"+"{}_mask.bmp".format(name[0:-4]))
            u_time = time.time()
            img_time = u_time-s_time
            print("image:{} time: {}".format(name, img_time))
            # print("type(grid) is:", type(grid_image))
            #print("grid_image.shape is:",grid_image.shape)
        print ("image save in out_path.")
        self.get_2type_img(args.out_path)
        # return("image save in out_path.")


    def get_2type_img(self,root_path):
        root=root_path
        dirlist = [ item for item in os.listdir(root) if os.path.isfile(os.path.join(root, item)) ]
        # print(dirlist)
        savePath='D:/Bio_hw/count_cell/cell_counter/image/two_type_mask/'

        for d in dirlist:
            img_path = root + '/'+d
            # print(img_path)
            img = cv2.imread(img_path)
            h,w,c = img.shape
            the_1 = np.zeros([h,w,3])
            the_2 = np.zeros([h,w,3])
            # monocyte
            # neutrophil

            for i in range(h):
                for j in range(w):
                    color = img[i,j] #color[0]=b, color[1]=g, color[2]=r
                    
                    if color[2] == 0 and color[1] == 128:
                        # print(img[i,j])
                        the_1[i,j] =[color[0],color[1],color[2]]
                    if color[2] == 128 and color[1] == 128:
                        the_2[i,j] =[color[0],color[1],color[2]]


            cv2.imwrite(savePath+d.split('.')[0]+'_monocyte.bmp', the_1)
            # monocyte = cv2.imread(d+'_monocyte.jpg')
            # img_gray1 = cv2.cvtColor(monocyte,cv2.COLOR_BGR2GRAY)

            cv2.imwrite(savePath+d.split('.')[0]+'_neutrophil.bmp', the_2)
            name=d.split('.')[0]
            # neutrophil = cv2.imread(d+'_neutrophil.jpg')
            # img_gray2 = cv2.cvtColor(neutrophil,cv2.COLOR_BGR2GRAY)
            monocyte, monocyte_img=self.get_connectedcomponent_count(savePath+d.split('.')[0]+'_monocyte.bmp',name+'_monocyte')
            neutrophil, neutrophil_img=self.get_connectedcomponent_count(savePath+d.split('.')[0]+'_neutrophil.bmp',name+'_neutrophil')

            if monocyte > neutrophil:
                final_calculat = monocyte/(monocyte+neutrophil)
                self.finalING_control(monocyte_img,final_calculat,name)
                
            elif neutrophil > monocyte :
                final_calculat = neutrophil/(monocyte+neutrophil)
                self.finalING_LPS(neutrophil_img,final_calculat,name)

            # return(final_calculat,name)
    
    def get_connectedcomponent_count(self,src_path,name):
        seve_path ='D:/Bio_hw/count_cell/cell_counter/image/counting_cells/'
        src = cv2.imread(src_path)

        src = cv2.GaussianBlur(src, (3, 3), 0)  #高斯模糊
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)#二值化
        # cv2.imshow("binary", binary)

        #主要API
        num_labels, labels, stats, centers = cv2.connectedComponentsWithStats(binary, connectivity=8, ltype=cv2.CV_32S)

        colors = []
        #生成隨機顏色
        for i in range(num_labels):
            b = np.random.randint(0, 256)
            g = np.random.randint(0, 256)
            r = np.random.randint(0, 256)
            colors.append((b, g, r))

        colors[0] = (0, 0, 0)
        image = np.copy(src)
        for t in range(1, num_labels, 1):
            x, y, w, h, area = stats[t]
            cx, cy = centers[t]

            #繪製圖像
            cv2.circle(image, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0)
            cv2.rectangle(image, (x, y), (x+w, y+h), colors[t], 1, 8, 0)

            cv2.putText(image, "num:" + str(t), (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            print("label index %d, area of the label : %d"%(t, area))

        # cv2.imshow("colored labels", image)
        cv2.imwrite(seve_path+name+'_after_count.bmp', image)
        img =cv2.imread(seve_path+name+'_after_count.bmp',cv2.IMREAD_UNCHANGED)
        print("total rice : ", num_labels - 1)
        return(num_labels - 1,img)

    def finalING_control(self,image,final_calculat,name):
        cv2.putText(image, "Control" , (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        cv2.putText(image, "monocyte >"+ str(round(final_calculat, 2)+"%") , (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        cv2.imwrite('D:/Bio_hw/count_cell/cell_counter/image/final_result/'+name+'final.bmp', image)
    def finalING_LPS(self,image,final_calculat,name):
        cv2.putText(image, "LPS" , (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        cv2.putText(image, "neutrophil >"+ str(round(final_calculat, 2)+"%") , (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1)
        cv2.imwrite('D:/Bio_hw/count_cell/cell_counter/image/final_result/'+name+'final.bmp', image)
 