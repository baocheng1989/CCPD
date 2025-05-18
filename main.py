import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import torch
import copy
import torchvision

from utils.cv_puttext import cv2ImgAddText
from plate_recognition.plate_rec import get_plate_result,allFilePath,init_model,cv_imread
from plate_recognition.double_plate_split_merge import get_split_merge
from models.experimental import attempt_load

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colors=[(0,255,0),(255,0,0),(0,0,255),(255,255,0),(0,255,255)]

# Helper functions from detector_ui.py would be placed here

def letter_box(img,size=(416,416)):
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

def load_model(weights, device):
    model = attempt_load(weights, device=device)
    return model

def xywh2rect(det):
    y = det.clone()
    y[:,0]=det[:,0]-det[0:,2]/2
    y[:,1]=det[:,1]-det[0:,3]/2
    y[:,2]=det[:,0]+det[0:,2]/2
    y[:,3]=det[:,1]+det[0:,3]/2
    return y

def my_nums(dets,iou_thresh):
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
        nuion_area1 =(y_box_score[i,2]-y_box_score[i,0])*(y_box_score[i,3]-y_box_score[i,1])
        union_area2 =(y_box_score[index[1:],2]-y_box_score[index[1:],0])*(y_box_score[index[1:],3]-y_box_score[index[1:],1])
        iou = inter_area/(nuion_area1+union_area2-inter_area)
        idx = torch.where(iou<=iou_thresh)[0]
        index=index[idx+1]
    return keep

def restore_box(dets,r,left,top):
    dets[:,[0,2]]=dets[:,[0,2]]-left
    dets[:,[1,3]]= dets[:,[1,3]]-top
    dets[:,:4]/=r
    return dets

def post_processing(prediction,conf,iou_thresh,r,left,top):
    xi = prediction[:,:,4]>conf
    x = prediction[xi]
    x[:,5:]*=x[:,4:5]
    boxes = x[:,:4]
    if boxes.size()[0]<1:
        return ""
    boxes = xywh2rect(boxes)
    score,index = torch.max(x[:,5:],dim=-1,keepdim=True)
    x = torch.cat((boxes,score,x[:,5:],index),dim=1)
    score = x[:,4]
    keep =my_nums(x,iou_thresh)
    x=x[keep]
    x=restore_box(x,r,left,top)
    return x

def pre_processing(img,device):
    img, r,left,top= letter_box(img,(416,416))
    img=img[:,:,::-1].transpose((2,0,1)).copy()
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img = img/255.0
    img =img.unsqueeze(0)
    return img ,r,left,top

def get_plate_result_all(img,detect_model,plate_rec_model,is_color=True):
    result_list = []
    im0 = copy.deepcopy(img)
    img,r,left,top = pre_processing(img,device)
    predict = detect_model(img)[0]
    outputs=post_processing(predict,0.3,0.5,r,left,top)
    if len(outputs):
        for output in outputs:
            one_list=[]
            output = output.squeeze().cpu().numpy().tolist()
            rect=output[:4]
            conf = output[4]
            rect = [int(x) for x in rect]
            label = int(output[-1])
            roi_img = im0[rect[1]:rect[3],rect[0]:rect[2]]
            if label:
                roi_img = get_split_merge(roi_img)
            plate_no,_,plate_color_,_ = get_plate_result(roi_img,device,plate_rec_model,is_color=is_color)
            one_list.append(plate_no+" "+plate_color_)
            one_list.append(conf)
            one_list.append(int(output[-1]))
            one_list.append(rect)
            result_list.append(one_list)
    return result_list

def init_model_detect(model_path,device):
    detect_model = load_model(model_path, device)
    detect_model.eval()
    return detect_model

def draw_result(orgimg,outputs,mainwin):
    result_str =""
    for output in outputs:
        rect = output[3]
        label = output[2]
        rect_area =rect
        cv2.rectangle(orgimg,(int(rect[0]),int(rect[1])),(int(rect[2]),int(rect[3])),colors[output[2]],3)
        plate_no = output[0]
        labelSize = cv2.getTextSize(plate_no,cv2.FONT_HERSHEY_SIMPLEX,0.5,1)
        orgimg=cv2.rectangle(orgimg,(rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1]))),(int(rect_area[0]+round(1.4*labelSize[0][0])),rect_area[1]+labelSize[1]),(255,255,255),cv2.FILLED)
        orgimg=cv2ImgAddText(orgimg,plate_no,rect_area[0],int(rect_area[1]-round(1.6*labelSize[0][1])),(0,0,0),21)
        result_str+=plate_no+"\n"
    print(result_str)
    if hasattr(mainwin, 'result_text'):
        mainwin.result_text.delete('1.0', tk.END)
        mainwin.result_text.insert(tk.END, result_str)
    return orgimg

class Detector:
    def __init__(self, mainwin) -> None:
        self.mainwin = mainwin
        self.is_color = True
        self.img_size = 416
        self.save_path = 'result'
        self.prproviders= torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.colors= [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),(0,0,0),(255,255,255),(255,0,255)]

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
    
    def modelLoad(self, detect_model, rec_model):
        self.detect_model = init_model_detect(detect_model,self.prproviders)
        self.rec_model = init_model(self.prproviders,rec_model,is_color=True)
        
    # Additional Detector methods would be added here

class ImagePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOV5 Image Predictor")
        
        # Model paths
        self.detect_model = r'weights\\best_yolov5s.pt'
        self.rec_model = r'weights\\plate_rec_color.pth'
        self.model = Detector(self)
        self.model.modelLoad(self.detect_model, self.rec_model)
        
        # GUI Elements
        self.btn_open = tk.Button(root, text="Open Image", command=self.open_image)
        self.btn_predict = tk.Button(root, text="Predict", command=self.predict_image)
        self.btn_open.pack(pady=10)
        self.btn_predict.pack(pady=10)
        
        # Image display
        self.image_label = tk.Label(root)
        self.image_label.pack()
        
        # Result display
        self.result_text = tk.Text(root, height=10, width=50)
        self.result_text.pack(pady=10)
        
        self.source = None
        
    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path and file_path.split('.')[-1].lower() in IMG_FORMATS:
            self.source = file_path
            self.display_image(file_path)
    
    def display_image(self, file_path):
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img
    
    def predict_image(self):
        if self.source:
            self.model.run('img', self.source)
            # In a real implementation, you would display the prediction results
            # For simplicity, we're just showing the original image here
            self.display_image(self.source)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictor(root)
    root.mainloop()