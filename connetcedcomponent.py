import cv2 as cv
import numpy as np
import os

def get_connectedcomponent_count(path):
    for src in os.listdir(path):

        # src = cv.imread(r"C:\Users\cwlab913\Documents\HW1025\cell_project\result\resnet\S__52756636_Copy_mask.bmp")

        src = cv.GaussianBlur(src, (3, 3), 0)  #高斯模糊
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)#二值化
        cv.imshow("binary", binary)

        #主要API
        num_labels, labels, stats, centers = cv.connectedComponentsWithStats(binary, connectivity=8, ltype=cv.CV_32S)

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
            cv.circle(image, (np.int32(cx), np.int32(cy)), 2, (0, 255, 0), 2, 8, 0)
            cv.rectangle(image, (x, y), (x+w, y+h), colors[t], 1, 8, 0)

            cv.putText(image, "num:" + str(t), (x, y), cv.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            print("label index %d, area of the label : %d"%(t, area))

        cv.imshow("colored labels", image)
        print("total rice : ", num_labels - 1)

        cv.waitKey(0)
        cv.destroyAllWindows()