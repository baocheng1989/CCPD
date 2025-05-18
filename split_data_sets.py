import os

import torch
import shutil
import os
import random

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        # file_path = os.path.join(rootPath,temp)
        if os.path.isfile(os.path.join(rootPath,temp)):
            if not temp.endswith(".txt"):
                allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
            
            
if __name__ == "__main__":
    file_path =r"F:\train_val\new_val_data"
    save_path = r"F:\train_val\new_val_data1"
    val_ratio = 0.1
    file_list = []
    allFilePath(file_path,file_list)
    file_count = len(file_list)

    count=0
    new_count=0
    random.shuffle(file_list)
    val_list = file_list[:int(val_ratio*file_count)]
    
    for pic_ in val_list:
        txt_path = pic_.replace(".jpg",".txt")
        pic_name = os.path.basename(pic_)
        new_pic_ = os.path.join(save_path,pic_name)
        new_txt_ = new_pic_.replace(".jpg",".txt")
        
        shutil.move(pic_,new_pic_)
        try:
            shutil.move(txt_path,new_txt_)
        except:
            print(txt_path)
            
        print(pic_,new_pic_)