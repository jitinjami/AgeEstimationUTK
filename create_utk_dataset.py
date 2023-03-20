import argparse
import enum
from pathlib import Path
import shutil
import numpy as np
import os
import sys
from tqdm import tqdm
import pandas as pd
import cv2

def get_args():
    parser = argparse.ArgumentParser(description="This script creates database for training from the UTKFace dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", "-i", type=str, default= "UTK", help="Folder of UTKFace image directory")
    parser.add_argument("--output", "-o", type=str, default= "UTK_db", help="path to output database mat file")
    parser.add_argument("--train_ratio", type=float, default = 0.55, help="Ratio of dataset in train folder")
    parser.add_argument("--test_ratio", type=float, default = 0.25, help="Ratio of dataset in test folder, remaining will go to validation folder")
    args = parser.parse_args()
    return args

def detect_fix_copy(img_path, img_name, output_img_path):
    '''
    Fix corrupt images or avoid if not fixable
    '''
    try:
        img = cv2.imread(img_path)
        cv2.imwrite(output_img_path+img_name, img)
    except(IOError, SyntaxError) as e :
      print(e)
      print("Unable to load/write Image : {} . Image might be destroyed".format(img_path) )


def main():
    args = get_args()

    #Input image directory, usually "UTK"
    image_dir = Path(os.getcwd()+ "/" + args.input)

    image_dir.makedirs(parents=True, exist_ok=True)

    if not os.path.exists(image_dir):
        sys.exit("Image input directory doesn't exist")

    #Output image directory, usually "UTK_db"
    output_path = Path(os.getcwd()+ "/" + args.output)

    output_path.makedirs(parents=True, exist_ok=True)

    if not os.path.exists(output_path):
        sys.exit("Output directory doesn't exist")

    train_ratio = args.train_ratio
    test_ratio = args.test_ratio
    try:
        assert (train_ratio+test_ratio < 1)
    except AssertionError:
        sys.exit("Train and test ratio should add up to less than 1")
    
    val_ratio = 1 - (train_ratio+test_ratio)

    #Defining paths for train, test, and validation dataset folders
    train_path = str(output_path) +'/train/'
    test_path = str(output_path) +'/test/'
    val_path = str(output_path) +'/val/'

    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.makedirs(train_path)

    if os.path.exists(val_path):
        shutil.rmtree(val_path)
    os.makedirs(val_path)

    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.makedirs(test_path)

    #Getting list of valid file names
    allFileNames = os.listdir(image_dir)
    for path in allFileNames:
        if ".jpg" not in path:
            allFileNames.remove(path)

    np.random.seed(42)  
    np.random.shuffle(allFileNames)

    #Splitting file names into train, test, val
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                          [int(len(allFileNames)* (1 - (val_ratio + test_ratio))), 
                                                           int(len(allFileNames)* (1 - test_ratio))])
    
    #Giving each image file name a path in train, test, and val lists
    train_FilePaths = [str(image_dir) + '/'+ str(name) for name in train_FileNames.tolist()]
    val_FilePaths = [str(image_dir) +'/' + str(name) for name in val_FileNames.tolist()]
    test_FilePaths = [str(image_dir) +'/' + str(name) for name in test_FileNames.tolist()]

    #Fixing images with detect_fix_copy
    for i,_ in enumerate(tqdm(train_FilePaths)):
        detect_fix_copy(train_FilePaths[i], train_FileNames[i], train_path)

    for i,_ in enumerate(tqdm(val_FilePaths)):
        detect_fix_copy(val_FilePaths[i], val_FileNames[i], val_path)

    for i,_ in enumerate(tqdm(test_FilePaths)):
        detect_fix_copy(test_FilePaths[i], test_FileNames[i], test_path)

    #List of folders
    folder_list = [train_path, test_path, val_path]

    #CSV names for train, test and val
    csv_names = ['gt_train.csv', 'gt_test.csv', 'gt_valid.csv']

    #For loop through each folder: train, test and val
    for i,_ in enumerate(folder_list):
        #Empty dictionary with path and age that will go into the csv
        data = {'file_name' : [], 'real_age' : []}
        for file_name in os.listdir(folder_list[i]):
            if ".jpg" in file_name:
                img_name = folder_list[i] + file_name

                #Getting age value from the image file name
                age = file_name.split("_")[0]
                if int(age) <= 100:
                    data["file_name"].append(img_name)
                    data["real_age"].append(age)
        df = pd.DataFrame(data=data)
        
        #Saving as CSV
        df.to_csv(str(output_path)+ "/" + csv_names[i], index=False)

if __name__ == '__main__':
    main()

