import cv2
import numpy as np
import os
def get_2type_img(root_path):
	root=root_path
	dirlist = [ item for item in os.listdir(root) if os.path.isfile(os.path.join(root, item)) ]

	print(dirlist)
	for d in dirlist:
		img_path = root + d
		print(img_path)
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


		cv2.imwrite(d+'_monocyte.jpg', the_1)
		monocyte = cv2.imread(d+'_monocyte.jpg')
		img_gray1 = cv2.cvtColor(monocyte,cv2.COLOR_BGR2GRAY)

		cv2.imwrite(d+'_neutrophil.jpg', the_2)
		neutrophil = cv2.imread(d+'_neutrophil.jpg')
		img_gray2 = cv2.cvtColor(neutrophil,cv2.COLOR_BGR2GRAY)
		# cv2.imshow('2',img_gray1)
		# cv2.waitKey()


		# contours, hierarchy = cv2.findContours(img_gray1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #檢測輪廓
		# count=0  
		# for cont in contours:
		#     count+=1    #數量+1

		# cv2.drawContours(img,contours,-1,(0,0,255))
		# cv2.imshow('sa',img)
		# cv2.waitKey()
		# print(count)

# if __name__ == '__main__':
# 	root_path = r"D:\Bio_hw\count_cell\cell_project\getAddW\pic\S__52756636_Copy_mask.bmp"
# 	get_2type_img(root_path)