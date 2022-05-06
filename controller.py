
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog
from UI import Ui_MainWindow
import cv2
from PyQt5.QtGui import QImage, QPixmap
from demo import deeplabv3P_demo
from img_controller import img_controller
import os

class MainWindow_controller(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__() # in python3, super(Class, self).xxx = super().xxx
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setup_control()

    def setup_control(self):
        # TODO
        self.ui.label_img.setText('')
        self.ui.file_button.clicked.connect(self.open_file) 
        self.ui.folder_button.clicked.connect(self.open_folder)
        self.ui.btn_get_model_path.clicked.connect(self.get_model_file)
        self.ui.btn_get_seve_ptah.clicked.connect(self.get_save_file)
        self.ui.demo_exe.clicked.connect(self.demo_deeplab)
        # self.img_path = 'cat_small.jpg'
        self.file_path = ''
        self.img_controller = img_controller(img_path=self.file_path,
                                             label_img=self.ui.label_img,
                                             label_file_path=self.ui.label_file_name,
                                             label_ratio=self.ui.label_ratio)

        # self.ui.btn_open_file.clicked.connect(self.open_file)         
        self.ui.btn_zoom_in.clicked.connect(self.img_controller.set_zoom_in)
        self.ui.btn_zoom_out.clicked.connect(self.img_controller.set_zoom_out)
        self.ui.slider_zoom.valueChanged.connect(self.getslidervalue)
        self.model_file=''
        self.save_file=''
        self.ori_file=''

        

    def demo_deeplab(self):
        get_deeplab_result=deeplabv3P_demo(self.ori_file,self.save_file,self.model_name)
        get_deeplab_result.get_demo_result()
        # print(img_name,get_demo_result)
        

    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self, "Open file", "./") # start path        
        self.init_new_picture(filename)

    def init_new_picture(self, filename):
        self.ui.slider_zoom.setProperty("value", 50)
        self.img_controller.set_path(filename)        

    def getslidervalue(self):        
        self.img_controller.set_slider_value(self.ui.slider_zoom.value()+1)

    def get_model_file(self):
        model_folder_path = QFileDialog.getExistingDirectory(self,"Open folder","./")                 # start path
        print(model_folder_path)
        for name in os.listdir(model_folder_path):
            model_name = name
        self.model_name=model_folder_path+'/'+model_name

    def get_save_file(self):
        save_folder_path = QFileDialog.getExistingDirectory(self,"Open folder","./")                 # start path
        print(save_folder_path)
        self.save_file=save_folder_path


    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self,"Open folder","./")                 # start path
        print(folder_path)
        self.ui.show_folder_path.setText(folder_path)
        self.ori_file=folder_path

    # def display_img(self):
    #     self.img = cv2.imread(self.img_path)
    #     height, width, channel = self.img.shape
    #     bytesPerline = 3 * width
    #     self.qimg = QImage(self.img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
    #     self.qpixmap = QPixmap.fromImage(self.qimg)
    #     self.qpixmap_height = self.qpixmap.height()
    #     self.ui.label_img.setPixmap(QPixmap.fromImage(self.qimg))    

    # def set_zoom_in(self):
    #     self.qpixmap_height -= 100
    #     self.resize_image()

    # def set_zoom_out(self):
    #     self.qpixmap_height += 100
    #     self.resize_image()

    # def resize_image(self):
    #     scaled_pixmap = self.qpixmap.scaledToHeight(self.qpixmap_height)
    #     self.ui.label_img.setPixmap(scaled_pixmap)