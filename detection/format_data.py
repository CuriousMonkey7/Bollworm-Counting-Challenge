"""
ref: https://www.youtube.com/watch?v=NU9Xr_NYslo&t=95s
"""
# id_17bfdb30efc25664233b0fbf
import os
import ast
import cv2
import shutil
import numpy as np
import pandas as pd
from sklearn import model_selection
from tqdm import tqdm
from multiprocessing import Pool

RAW_DATA_DIR = "../data/raw/"
PREPROC_DATA_DIR = "../data/preproc/"

OUTPUT_DIR = "yolov5/detection/yolo_data/all/"


def process_row(row):
    img_id = row["image_id"].split(".")[0]
    split = row["split"]
    try:
        img_h,img_w,_ =cv2.imread(os.path.join(RAW_DATA_DIR,"images",f"{img_id}.jpg")).shape
    except Exception as e:
        print(img_id)
        return
        # raise (e)
    class_wise_bbox = row["bbox"]
    yolo_data = []

    for class_type,bboxes in class_wise_bbox.items():
        class_idx = 0 if class_type =="abw" else 1
        for points in bboxes:
            points = np.array(points,dtype= int)
            points =np.array([[points[i],points[i+1]] for i in range(0,len(points)-1,2)])
            x_min,y_min,W,H = cv2.boundingRect(points)
            x_center =x_min+W/2
            y_center = y_min+H/2 
            x_center /= img_w 
            y_center /=img_h
            W /=img_w
            H /=img_h
            yolo_data.append([class_idx,x_center,y_center,W,H])
    
    yolo_data = np.array(yolo_data)
    label_path = os.path.join(OUTPUT_DIR,"labels",split)
    os.makedirs(label_path,exist_ok=True)
    np.savetxt(
        os.path.join(label_path,f"{img_id}.txt"),
        yolo_data,
        fmt=["%d","%f","%f","%f","%f"]
        )
    img_path = os.path.join(OUTPUT_DIR,"images",split)
    os.makedirs(img_path,exist_ok=True)
    shutil.copy(
        os.path.join(PREPROC_DATA_DIR,"jpg",f"{img_id}.jpg"),
        os.path.join(img_path,f"{img_id}.jpg")
    )
        

def process_data(df):
    with Pool() as p:
        it =df.iterrows()
        temp =[]
        for i,row in it:
            temp.append(row)
        p.map(process_row, temp)   
    


if __name__ =="__main__":
    data_df = pd.read_csv("yolov5/detection/Train.csv")
    data_df.bbox = data_df.bbox.apply(ast.literal_eval)
    n_folds = 5
    df_train,df_valid = model_selection.train_test_split(test_size=0.1, shuffle=True, random_state=42)
    # for fold, (train_idx, val_idx) in enumerate(kf.split(data_df)):
    #     data_df.loc[val_idx, 'fold'] = fold

    # # df_test = 
    # for fold in range(n_folds):
    #     print("fold",fold)
    #     df_train = data_df.query("fold!=@fold").reset_index(drop=True)
    #     df_valid = data_df.query("fold==@fold").reset_index(drop=True)
    #     df_train["split"] = "train"
    #     df_valid["split"] = "validation"
    #     df_train["fold"] = fold
    process_data(df_train)
    process_data(df_valid)
    
    
    """Test"""
    # test = pd.read_csv("../data/preds/test_droupout0.3_filter_preds.csv")
    # test = test[test["preds"]!=0]
    # print(len(test))
    # test["split"] = "test"
    
    # test.columns = ["image_id","preds","split"]
    # process_data(test)