import cv2
import os
import matplotlib.pyplot as plt
path = 'D:/Bio_hw/count_cell/cell_counter/image/final_result/'
path2 = 'D:/Bio_hw/count_cell/cell_counter/image/original_Image/'
path3 = 'D:/Bio_hw/count_cell/cell_counter/image/get_add_weight/'
dirlist = os.listdir(path)

for i in dirlist:

    # print (i,len(i))
    if len(i) <=40 :
        os.chdir(path)
        # print("Current working directory: {0}".format(os.getcwd()))
        img1=cv2.imread(i)
        # cv2.imshow('img1',img1)
        os.chdir(path2)
        img2=cv2.imread(i[:-14]+'.jpg')
        print(i[:-14]+'.jpg')
        # cv2.imshow('rgb',rgb)
        rgb=img2[...,::-1]
        dst=cv2.addWeighted(img1,0.5,rgb,0.5,0)
        # cv2.imshow('dst',dst)
        dst_rgb=dst[...,::-1]

        # save  img
        os.chdir(path3)
        cv2.imwrite(i[:-4]+'_addWeighted'+'.bmp',dst)


        fig, axes = plt.subplots(nrows=2, ncols=2,
                         sharex=True, sharey=True, figsize=(10, 10))
        ax = axes.ravel()

        ax[0].imshow(img1, cmap=plt.cm.gray)
        ax[0].set_title('Original image')
        ax[1].imshow(rgb, cmap=plt.cm.gray)
        ax[1].set_title('Mask image, radius=1, amount=1.0')
        ax[2].imshow(dst_rgb, cmap=plt.cm.gray)
        ax[2].set_title('dst image, radius=5, amount=2.0')
        for a in ax:
            a.axis('off')
        fig.tight_layout()
        plt.show()


        cv2.waitKey(0)