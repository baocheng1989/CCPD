import torch
import cv2
import numpy as np
from models.experimental import attempt_load
import argparse
import copy
import torchvision
import os
from utils.cv_puttext import cv2ImgAddText
# from plate_recognition.plate_rec import allFilePath, cv_imread
from plate_recognition.plate_rec import get_plate_result,allFilePath,init_model,cv_imread
from plate_recognition.double_plate_split_merge import  get_split_merge
colors=[(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255)]
def letter_box(img,size=(640,640)):#yolov5 前处理 图片缩放到640X640
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

def pre_processing(img,device,img_size):  #前处理
    img, r,left,top= letter_box(img,img_size)
    # print(img.shape)
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

def get_plate_result_all(img,detect_model,plate_rec_model,img_size,is_color=True):
    result_list = []
    im0 = copy.deepcopy(img)
    img,r,left,top = pre_processing(img,device,img_size)  #检测前处理
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
            
            plate_no,_,plate_color,_ = get_plate_result(roi_img,device,plate_rec_model,is_color=is_color) #对车牌小图区域进行识别得到车牌号
            one_list.append(plate_no+" "+plate_color)
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



def draw_result(orgimg,outputs):   # 车牌结果画出来
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
        result_str+=plate_no+" "
    print(result_str)
    return orgimg
 
    

if __name__ == "__main__":
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--detect_model', nargs='+', type=str, default=r'weights\best_yolov5s_final.pt', help='model.pt path(s)')  #yolov5检测模型
    parser.add_argument('--rec_model', type=str, default='weights/plate_rec_color.pth', help='model.pt path(s)')#车牌识别模型
    parser.add_argument('--image_path', type=str, default=r'imgs', help='source')   #待识别图片路径
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)') #识别后结果保存地址
    parser.add_argument('--output', type=str, default='result', help='source') 
  
    

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    opt = parser.parse_args()
    if not os.path.exists(opt.output):
        os.mkdir(opt.output)
    device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detect_model = init_model_detect(opt.detect_model,device)
    plate_rec_model=init_model(device,opt.rec_model,is_color=True)
    
    # print(detect_model)
    file_list = []
    allFilePath(opt.image_path,file_list)
    count= 0
    time_all = 0
    time_begin=time.time()
    for pic_ in file_list:
        print(count,pic_)
        # count+=1
        img = cv2.imread(pic_)
        im0 = copy.deepcopy(img)
        time_b = time.time()  
        outputs=get_plate_result_all(img,detect_model,plate_rec_model,(opt.img_size,opt.img_size))
        time_e=time.time()
        time_gap=time_e-time_b
        if count:
            time_all+=time_gap 
        count+=1
        if len(outputs):
            im0 = draw_result(im0,outputs)
            image_name = os.path.basename(pic_)
            new_image_path =os.path.join(opt.output,image_name)
        
        cv2.imwrite(new_image_path,im0)
    print(f"sumTime time is {time.time()-time_begin} s, average pic time is {time_all/(len(file_list)-1)*1000} ms")