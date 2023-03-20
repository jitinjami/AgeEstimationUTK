import gdown
import tarfile
import os
import shutil
import zipfile

#Download and Save all UTK crop dataset images as zip file from GDrive
url = 'https://drive.google.com/uc?id=1ATmg1HZ3BT0hfobwFhiO86PV5ty3rCrR'
output = 'utkcropped.zip'
gdown.download(url, output, quiet=False)

#Extract zip file
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall("./")

#Move all images to "UTK" directory
pwd = os.getcwd()

destination = pwd + "/UTK/"

source = pwd + "/utkcropped/"

allfiles = os.listdir(source)

for f in allfiles:
    src_path = os.path.join(source, f)
    dst_path = os.path.join(destination, f)
    shutil.move(src_path, dst_path)