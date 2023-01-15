# import glob
# import cv2
# import warnings
# from multiprocessing import Pool

# warnings.filterwarnings("error")

# files = glob.glob("yolov5/detection/yolo_data/images/train/*")
# def check(file):
#     try:
#         cv2.imread(file) 
#     except:
#         print(file)

# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
#     with Pool() as p:
#         res = p.map(check,files)
#     print(res) 


import glob
import cv2
import warnings
import numpy as np
warnings.filterwarnings("error")
from tqdm.auto import tqdm
files = glob.glob("yolov5/detection/yolo_data/images/train/*")
corrupted_image = []
for file in tqdm(files):
    try:
        cv2.imread(file) 
    except:
        corrupted_image.append(file)
        
np.savetxt("./corrupted_image.txt",
    np.array(corrupted_image))
    