import gdown
import tarfile
import os
import shutil

#Download and Save all UTK dataset images as zip files from GDrive
url = 'https://drive.google.com/uc?id=1mb5Z24TsnKI3ygNIlX6ZFiwUj0_PmpAW'
output = 'part1.tar.gz'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/uc?id=19vdaXVRtkP-nyxz1MYwXiFsh_m_OL72b'
output = 'part2.tar.gz'
gdown.download(url, output, quiet=False)


url = 'https://drive.google.com/uc?id=1oj9ZWsLV2-k2idoW_nRSrLQLUP3hus3b'
output = 'part3.tar.gz'
gdown.download(url, output, quiet=False)

#Extract all zip files
file = tarfile.open("part1.tar.gz")

file.extractall('./')
  
file.close()

file = tarfile.open("part2.tar.gz")

file.extractall('./')
  
file.close()

file = tarfile.open("part3.tar.gz")

file.extractall('./')
  
file.close()

#Move all images to "UTK" directory
pwd = os.getcwd()

destination = pwd + "/UTK/"

source = pwd + "/part1/"

allfiles = os.listdir(source)

for f in allfiles:
    src_path = os.path.join(source, f)
    dst_path = os.path.join(destination, f)
    shutil.move(src_path, dst_path)

source = pwd + "/part2/"

allfiles = os.listdir(source)
 
for f in allfiles:
    src_path = os.path.join(source, f)
    dst_path = os.path.join(destination, f)
    shutil.move(src_path, dst_path)

source = pwd + "/part3/"

allfiles = os.listdir(source)
 
for f in allfiles:
    src_path = os.path.join(source, f)
    dst_path = os.path.join(destination, f)
    shutil.move(src_path, dst_path)