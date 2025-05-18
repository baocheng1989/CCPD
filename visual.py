import traceback
from UI.ui import Ui_MainWindow
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
import detector_ui
import os
import numpy as np
import time

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
VID_FORMATS = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes
# 预测线程, 调用yolo模型进行推理
class PredictThread(QThread):
    # 自定义信号对象。参数str就代表这个信号可以传一个字符串
    trigger = pyqtSignal(int)

    def __init__(self, mainwin):
        # 初始化函数
        # super().__init__()
        super(PredictThread, self).__init__()
        self.mainwin = mainwin

    def run(self):
        #重写线程执行的run函数
        try:
            self.mainwin.model_class.run(self.mainwin.mode, self.mainwin.source)
            #触发自定义信号
            self.trigger.emit(1)
            # self.mainwin.model_class.run(self.mainwin.source, False, '')
        except Exception as e:
            # print('ERROR: %s' %(e))
            print(traceback.print_exc())

class MainWin(Ui_MainWindow, QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("YOLOV5 可视化接口")
        desktop = QApplication.desktop()
        min_ratio = 0.5
        max_ratio = 0.8
        self.setGeometry(int((1-max_ratio)*desktop.width()/2), int((1-max_ratio)*desktop.height()/2), int(max_ratio*desktop.width()), int(max_ratio*desktop.height()))

        self.PB_import.clicked.connect(self.importMedia) # 打开按钮
        self.PB_predict.clicked.connect(self.run) # 预测按钮
        self.PB_predict.setEnabled(True) 
        self.PB_stop.clicked.connect(self.stopPredict) # 停止按钮
        self.PB_resize.clicked.connect(self.resize_label) # 适应窗口按钮
        self.stop = 0 # 视频/摄像头时起作用, 当stop == 1时, 视频/摄像头停止播放

        self.canrun = 0
        self.source = None
        self.detect_model = r'weights\best_yolov5s.pt'
        self.rec_model = r'weights\plate_rec_color.pth'

        self.model_class = detector_ui.Detector(self)  # yolov5模型
        self.loadmodel()

        self.predictThread = PredictThread(self)  # 初始化预测线程
        self.predictThread.trigger.connect(self.isdone)

        # self.updataListWidgetin()
        self.timer=QTimer()
        self.timer.timeout.connect(self.showtime)#这个通过调用槽函数来刷新时间
        self.timer.start(1000)

        self.t_init = int(time.time())
        self.L_day.setAlignment(Qt.AlignHCenter)
        self.L_time.setAlignment(Qt.AlignHCenter)
        self.L_during.setAlignment(Qt.AlignHCenter)

    def showtime(self):
        t = time.localtime()
        self.L_day.setText(f'{t.tm_year}年{t.tm_mon}月{t.tm_mday}日')
        self.L_time.setText(f'{str(t.tm_hour).zfill(2)}:{str(t.tm_min).zfill(2)}:{str(t.tm_sec).zfill(2)}')

        t_now = int(time.time())
        m, s = divmod(t_now - self.t_init, 60)
        h, m = divmod(m, 60)
        self.L_during.setText(f'{str(h).zfill(2)}:{str(m).zfill(2)}:{str(s).zfill(2)}')

    def updataListWidgetin(self):
        self.listWidget_in.clear()
        self.listWidget_in.addItem('输入信息:')
        self.listWidget_in.item(0).setBackground(QColor(0,255,255))
        self.listWidget_in.addItem('权重名称:')
        self.listWidget_in.addItem('    ' + os.path.basename(self.detect_model))
        self.listWidget_in.addItem('    ' + os.path.basename(self.rec_model))
        self.listWidget_in.addItem('数据源:')
        try:
            self.listWidget_in.addItem('    ' + os.path.basename(self.source))
        except:
            self.listWidget_in.addItem('    ' + '摄像头')
    
    def resizeImg(self, image):
        '''
        调整图片到合适大小
        '''
        width = image.width()  ##获取图片宽度
        height = image.height() ##获取图片高度
        if width / self.labelsize[1] >= height / self.labelsize[0]: ##比较图片宽度与label宽度之比和图片高度与label高度之比
            ratio = width / self.labelsize[1]
        else:
            ratio = height / self.labelsize[0]
        new_width = width / ratio  ##定义新图片的宽和高
        new_height = height / ratio
        new_img = image.scaled(new_width, new_height)##调整图片尺寸
        return new_img

    def padding(self, image):
        '''
        图片周围补0以适应label大小
        '''
        width = image.shape[1]
        height = image.shape[0]
        target_ratio = self.labelsize[0]/self.labelsize[1] # h/w
        now_ratio = height/width
        if target_ratio>now_ratio:
            # padding h
            new_h = int(target_ratio*width)
            padding_image = np.ones([int((new_h-height)/2), width, 3], np.uint8)*255
            new_img = cv2.vconcat([padding_image, image, padding_image])
        else:
            # padding w
            new_w = int(height/target_ratio)
            padding_image = np.ones([height, int((new_w-width)/2), 3], np.uint8)*255
            new_img = cv2.hconcat([padding_image, image, padding_image])
        return new_img

    def resize_label(self):
        '''
        更新label中的图片大小
        '''
        self.labelsize = [self.label_in.height(), self.label_in.width()]
        img_in = self.label_in.pixmap()
        img_out = self.label_out.pixmap()
        try:
            img_in = self.resizeImg(img_in)
        except:
            return
        else:
            self.label_in.setPixmap(img_in)

        try:
            img_out = self.resizeImg(img_out)
        except:
            return
        else:    
            self.label_out.setPixmap(img_out)

    def importMedia(self):
        '''
        打开检测源
        '''
        self.labelsize = [self.label_out.height(), self.label_out.width()]
        # 源为摄像头
        if self.RB_camera.isChecked():
            self.mode = 'vid'
            self.source = 0
            self.updataListWidgetin()
            self.run()
            # print('<font color=green>请载入模型进行预测...</font>')
        # 源为图片/视频
        elif self.RB_img.isChecked():
            fname, _ = QFileDialog.getOpenFileName(self, "打开文件", ".")
            # print(fname)
            if fname.split('.')[-1].lower() in (IMG_FORMATS + VID_FORMATS):
                self.source = fname          
                self.updataListWidgetin()
                self.importImg(fname)
            else:
                print('<font color=red>不支持该类型文件...</font>')
        else:
            print('<font color=red>请选择检测源类型...</font>')
    
    def importImg(self, file_name):
        '''
        label_in 中显示图片/视频第一帧
        '''
        if file_name.split('.')[-1].lower() in VID_FORMATS:
            self.mode = 'vid'
            cap = cv2.VideoCapture(file_name)
            if cap.isOpened():
                # self.video = True
                ret, img_in = cap.read()
                if ret:
                    img_in = self.padding(img_in)
                    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB)
                    # padding
                    img_in = QImage(img_in, img_in.shape[1], img_in.shape[0], QImage.Format_RGB888)
                    img_in = QPixmap(img_in)
            cap.release()
        elif file_name.split('.')[-1].lower() in IMG_FORMATS:
            self.mode = 'img'
            # self.video = False
            img_in = cv2.imread(file_name)
            img_in = self.padding(img_in)
            img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGBA)
            img_in = QImage(img_in, img_in.shape[1], img_in.shape[0], QImage.Format_RGBA8888)
            img_in = QPixmap(img_in)
        if img_in.isNull():
            print('<font color=red>打开失败...</font>')
            return
        # img_in = img_ni.scaledToWidth(self.labelsize[1])
        img_in = self.resizeImg(img_in)
        self.label_in.setPixmap(img_in)

    def stopPredict(self):
        '''
        stop == 1 播放停止
        '''
        self.stop = 1

    def loadmodel(self):
        '''
        载入模型
        '''
        self.canrun = 1
        self.model_class.modelLoad(self.detect_model, self.rec_model)

    def run(self):
        '''
        开始预测
        '''
        if self.canrun == 0:
            print('<font color=red>请载入模型...</font>')
            return
        elif self.source == None:
            print('<font color=red>请选择检测源...</font>')
            return
        # elif self.save_img == True and self.save_path == '':
        #     print('<font color=red>请选择保存路径...</font>')
        else:
            self.predictThread.start()
            self.canrun = 0
            self.PB_predict.setEnabled(False)
            self.action_loadmodel.setEnabled(False)
            self.PB_stop.setEnabled(True)

    def isdone(self, done):
        '''
        结束一次预测
        '''
        if done == 1:
            self.canrun = 1
            self.PB_predict.setEnabled(True)
            self.action_loadmodel.setEnabled(True)
            # self.PB_import.setEnabled(True)
            self.PB_stop.setEnabled(False)
            self.stop = 0
            self.predictThread.quit()




if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWin()
    main.show()
    sys.exit(app.exec_())
