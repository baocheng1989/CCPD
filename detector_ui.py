# import onnxruntime
import torch
import numpy as np
import cv2
import copy
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from utils.cv_puttext import cv2ImgAddText
from plate_recognition.plate_rec import get_plate_result,allFilePath,init_model,cv_imread
from plate_recognition.double_plate_split_merge import  get_split_merge
import torch
import cv2
import numpy as np
from models.experimental import attempt_load
import argparse
import copy
import torchvision
import os
from utils.cv_puttext import cv2ImgAddText

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors=[(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255)]

def letter_box(img,size=(416,416)):#yolov5 前处理 图片缩放到416X416
    h,w,_=img.shape
    r=min(size[0]/h,size[1]/w)
    new_h,new_w=int(h*r),int(w*r)
    new_img = cv2.resize(img,(new_w,new_h))
    left= int((size[1]-new_w)/2)
    top=int((size[0]-new_h)/2)   
    right = size[1]-left-new_w
    bottom=size[0]-top-new_h 
    img =cv2.copyMakeBorder(new_img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
    return img,r,left,top

def load_model(weights, device):   #导入yolov5模型
    model = attempt_load(weights, device=device)  # load FP32 model
    return model    

def xywh2rect(det):         #yolo格式转为rect模型
    y = det.clone()
    y[:,0]=det[:,0]-det[0:,2]/2
    y[:,1]=det[:,1]-det[0:,3]/2
    y[:,2]=det[:,0]+det[0:,2]/2
    y[:,3]=det[:,1]+det[0:,3]/2
    return y
 
def my_nums(dets,iou_thresh):   #yolo中用到的nms
    y = dets.clone()
    y_box_score = y[:,:5]
    index = torch.argsort(y_box_score[:,-1],descending=True)
    keep = []
    while index.size()[0]>0:
        i = index[0].item()
        keep.append(i)
        x1=torch.maximum(y_box_score[i,0],y_box_score[index[1:],0])
        y1=torch.maximum(y_box_score[i,1],y_box_score[index[1:],1])
        x2=torch.minimum(y_box_score[i,2],y_box_score[index[1:],2])
        y2=torch.minimum(y_box_score[i,3],y_box_score[index[1:],3])
        zero_=torch.tensor(0).to(device)
        w=torch.maximum(zero_,x2-x1)
        h=torch.maximum(zero_,y2-y1)
        inter_area = w*h
        nuion_area1 =(y_box_score[i,2]-y_box_score[i,0])*(y_box_score[i,3]-y_box_score[i,1]) #计算交集
        union_area2 =(y_box_score[index[1:],2]-y_box_score[index[1:],0])*(y_box_score[index[1:],3]-y_box_score[index[1:],1])#计算并集

        iou = inter_area/(nuion_area1+union_area2-inter_area)#计算iou
        
        idx = torch.where(iou<=iou_thresh)[0]   #保留iou小于iou_thresh的
        index=index[idx+1]
    return keep

def restore_box(dets,r,left,top):  #坐标还原到原图上

    dets[:,[0,2]]=dets[:,[0,2]]-left
    dets[:,[1,3,]]= dets[:,[1,3]]-top
    dets[:,:4]/=r
    # dets[:,5:13]/=r

    return dets
    # pass

def post_processing(prediction,conf,iou_thresh,r,left,top):  #后处理
    xi = prediction[:,:,4]>conf 
    x = prediction[xi]          #过滤掉小于conf的框
    x[:,5:]*=x[:,4:5]          #得分为object_score*class_score
    boxes = x[:,:4]
    if boxes.size()[0]<1:
        return ""
    boxes = xywh2rect(boxes)  #中心点 宽高 变为 左上 右下两个点
    score,index = torch.max(x[:,5:],dim=-1,keepdim=True) #找出类别和得分
    x = torch.cat((boxes,score,x[:,5:],index),dim=1)      #重新组合
    score = x[:,4]
    # i = torchvision.ops.nms(boxes, score, iou_thresh)
    keep =my_nums(x,iou_thresh)
    x=x[keep]
    x=restore_box(x,r,left,top)
    return x

def pre_processing(img,device):  #前处理
    img, r,left,top= letter_box(img,(416,416))
    print(img.shape)
    img=img[:,:,::-1].transpose((2,0,1)).copy()  #bgr2rgb hwc2chw
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img = img/255.0
    img =img.unsqueeze(0)
    return img ,r,left,top

# device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
# detect_model = load_model(r"weights\best_7.0.pt", device)
# plate_rec_model=init_model(device,r"weights/plate_rec.pth")
# detect_model.eval() 

def get_plate_result_all(img,detect_model,plate_rec_model,is_color=True):
    result_list = []
    im0 = copy.deepcopy(img)
    img,r,left,top = pre_processing(img,device)  #检测前处理
    predict = detect_model(img)[0]                   
    outputs=post_processing(predict,0.3,0.5,r,left,top) #检测后处理
    if len(outputs): 
        for output in outputs:
            one_list=[]
            output = output.squeeze().cpu().numpy().tolist()
            rect=output[:4]              #车牌坐标
            conf = output[4]             #车牌得分
            rect = [int(x) for x in rect]
            label = int(output[-1])
            roi_img = im0[rect[1]:rect[3],rect[0]:rect[2]] #车牌区域小图
            if label:
                roi_img = get_split_merge(roi_img)
            # land_marks=output[5:13]
            # roi_img = im0[rect[1]:rect[3],rect[0]:rect[2]] #车牌区域小图
            plate_no,_,plate_color_,_ = get_plate_result(roi_img,device,plate_rec_model,is_color=is_color) #对车牌小图区域进行识别得到车牌号
            one_list.append(plate_no+" "+plate_color_)
            one_list.append(conf)
            
            one_list.append(int(output[-1]))
            one_list.append(rect)
            result_list.append(one_list)
            # height_area = roi_img.shape[0] #  车牌区域图片的高度
            # im0=cv2ImgAddText(im0,plate_no,rect[0]-height_area,rect[1]-height_area-10,(0,255,0),height_area)#将车牌结果画在原图上
            # cv2.imwrite("haha.jpg",roi_img)
            # c = int(cls)  # integer class
            # cv2.rectangle(im0,(rect[0],rect[1]),(rect[2],rect[3]),(0,255,255),2)#黄框
    return result_list

def init_model_detect(model_path,device):
    detect_model = load_model(model_path, device)
    # plate_rec_model=init_model(device,r"weights/plate_rec.pth")
    detect_model.eval() 
    return detect_model



def draw_result(orgimg,outputs,mainwin):   # 车牌结果画出来
    result_str =""

    for output in outputs:
        rect = output[3]
        label = output[2]
        rect_area =rect
        cv2.rectangle(orgimg,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),colors[output[2]],3)
        plate_no = output[0]
        # result_p = label_str[output['label']]
        # cv2.rectangle(orgimg,(rect_area[0],rect_area[1]),(rect_area[2],rect_area[3]),(0,0,255),2) #画框
        
        labelSize = cv2.getTextSize(plate_no,cv2.FONT_HERSHEY_SIMPLEX,0.5,1) #获得字体的大小
        # if rect_area[0]+labelSize[0][0]>orgimg.shape[1]:                 #防止显示的文字越界
        #     rect_area[0]=int(orgimg.shape[1]-labelSize[0][0])
        orgimg=cv2.rectangle(orgimg,(rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1]))),(int(rect_area[0]+round(1.4*labelSize[0][0])),rect_area[1]+labelSize[1]),(255,255,255),cv2.FILLED)#画文字框,背景白色
        
        # if len(result['plate_no'])>=3:
        # result_p='赣'+result_p[1:]
        orgimg=cv2ImgAddText(orgimg,plate_no,rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1])),(0,0,0),21)
        result_str+=plate_no+"\n"
    print(result_str)
    mainwin.listWidget_out.clear()
    mainwin.listWidget_out.addItem('输出信息:')
    mainwin.listWidget_out.item(0).setBackground(QColor(0,255,255))
    mainwin.listWidget_out.addItem(result_str)
    return orgimg
 

# def draw_result(orgimg,outputs,mainwin):   # 车牌结果画出来
#     result_str =""

#     for output in outputs:
#         rect = output[2]
#         label = label_str[output[1]]
#         rect_area =rect
#         cv2.rectangle(orgimg,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),colors[output[1]],5)
#         # result_p = label_str[output['label']]
#         # cv2.rectangle(orgimg,(rect_area[0],rect_area[1]),(rect_area[2],rect_area[3]),(0,0,255),2) #画框
        
#         labelSize = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.5,1) #获得字体的大小
#         # if rect_area[0]+labelSize[0][0]>orgimg.shape[1]:                 #防止显示的文字越界
#         #     rect_area[0]=int(orgimg.shape[1]-labelSize[0][0])
#         orgimg=cv2.rectangle(orgimg,(rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1]))),(int(rect_area[0]+round(1.4*labelSize[0][0])),rect_area[1]+labelSize[1]),(255,255,255),cv2.FILLED)#画文字框,背景白色
        
#         # if len(result['plate_no'])>=3:
#         # result_p='赣'+result_p[1:]
#         orgimg=cv2ImgAddText(orgimg,label,rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1])),(0,0,0),21)
#         result_str+=label+"\n"
#     print(result_str)
#     mainwin.listWidget_out.clear()
#     mainwin.listWidget_out.addItem('输出信息:')
#     mainwin.listWidget_out.item(0).setBackground(QColor(0,255,255))
#     mainwin.listWidget_out.addItem(result_str)
#     return orgimg




class Detector():
    def __init__(self, mainwin) -> None:
        self.mainwin = mainwin
        self.is_color = True
        # self.image_path = 'imgs'
        self.img_size = 416
        self.save_path = 'result'
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prproviders= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.colors= [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(0,0,0),(255,255,255),(255,0,255)]

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
    
    def modelLoad(self, detect_model, rec_model):
        self.detect_model = init_model_detect(detect_model,self.prproviders)
        self.rec_model = init_model(self.prproviders,rec_model,is_color=True)
        
    def displayImg_in(self, img):
        '''
        label_in 中显示输入图像
        '''
        img = self.mainwin.padding(img)
        RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_out = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGBA8888)
        img_out = QPixmap(img_out)
        # img_out = img_out.scaledToWidth(self.mainwin.labelsize[1])
        img_out = self.mainwin.resizeImg(img_out)
        self.mainwin.label_in.setPixmap(img_out)

    def displayImg_out(self, img):
        '''
        label_out 中显示预测结果
        '''
        img = self.mainwin.padding(img)
        RGBImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        img_out = QImage(RGBImg, RGBImg.shape[1], RGBImg.shape[0], QImage.Format_RGBA8888)
        img_out = QPixmap(img_out)
        # img_out = img_out.scaledToWidth(self.mainwin.labelsize[1])
        img_out = self.mainwin.resizeImg(img_out)
        self.mainwin.label_out.setPixmap(img_out)

    def run(self, mode, source):
        if mode == 'img':     #处理图片
            img =cv_imread(source)
            if img.shape[-1]==4:
                img=cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
            self.displayImg_in(img)
            
            
            # img=cv2.imread(pic_)
            img0 = copy.deepcopy(img)
            outputs=get_plate_result_all(img,self.detect_model,self.rec_model,is_color=True)
            # for output in outputs:
            #     # label =int(output[5])
            #     rect = output['rect']
            #     cv2.rectangle(img0,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),self.colors[output['label']],2)
            img0 = draw_result(img0,outputs, self.mainwin)
            
            
            # dict_list=detect_Recognition_plate(self.detect_model, img, self.device,self.plate_rec_model,self.img_size,is_color=self.is_color)
            # ori_img=draw_result(img,dict_list, self.mainwin)
            self.displayImg_out(img0)
            img_name = os.path.basename(source)
            save_img_path = os.path.join(self.save_path,img_name)
            # cv2.imwrite(save_img_path,img0)  
        else:    #处理视频
            capture=cv2.VideoCapture(source)
            fourcc = cv2.VideoWriter_fourcc(*'MP4V') 
            fps = capture.get(cv2.CAP_PROP_FPS)  # 帧数
            width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 宽高
            out = cv2.VideoWriter('result.mp4', fourcc, fps, (width, height))  # 写入视频
            frame_count = 0
            fps_all=0
            # rate,FrameNumber,duration=get_second(capture)
            if capture.isOpened():
                while True:
                    t1 = cv2.getTickCount()
                    frame_count+=1
                    print(f"第{frame_count} 帧",end=" ")
                    ret,img=capture.read()
                    if not ret:
                        break
                    if self.mainwin.stop == 1:
                        break
                    # if frame_count%rate==0:
                    self.displayImg_in(img)
                    img0 = copy.deepcopy(img)
                    outputs=get_plate_result_all(img,self.detect_model,self.rec_model)
            # for output in outputs:
            #     # label =int(output[5])
            #     rect = output['rect']
            #     cv2.rectangle(img0,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),self.colors[output['label']],2)
                    img0 = draw_result(img0,outputs, self.mainwin)
                    # ori_img=draw_result(img,dict_list, self.mainwin)
                    t2 =cv2.getTickCount()
                    infer_time =(t2-t1)/cv2.getTickFrequency()
                    fps=1.0/infer_time
                    fps_all+=fps
                    str_fps = f'fps:{fps:.4f}'
                    self.displayImg_out(img0)
                    cv2.putText(img0,str_fps,(20,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    # out.write(img0)
            else:
                print("失败")
            capture.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"all frame is {frame_count},average fps is {fps_all/frame_count} fps")

