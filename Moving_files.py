

import os
import random
import shutil
import math 

source = 'J:/No_label_spectro/E'
def percentage(percent, whole):
  return (percent * whole) / 100.0

#-------------------------------------------TRAINING SET--------------------------------------
dest = 'J:\SETS\SETS_OX\train'
files = os.listdir(source)
def percentage(percent, whole):
  return (percent * whole) / 100.0
percentage = percentage(60,len(files))
no_of_files = math.ceil(percentage)
print(no_of_files)
for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)
    
#------------------------------------------- VALIDATION SET--------------------------------------
dest = 'J:/New-Total-Aug-Set/Validate/E'
files = os.listdir(source)
def percentage(percent, whole):
  return (percent * whole) / 100.0
percentage = percentage(50,len(files))
no_of_files = math.ceil(percentage)
print(no_of_files)
for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)
    
#------------------------------------------- TESTING SET--------------------------------------
dest = 'J:/New-Total-Aug-Set/Test/E'
files = os.listdir(source)
def percentage(percent, whole):
  return (percent * whole) / 100.0
percentage = percentage(100,len(files))
no_of_files = math.ceil(percentage)
print(no_of_files)
for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)
    
    
    folder = 'J:/PNG_3SET/Augmented/E/'
count = 1
# count increase by 1 in each iteration
# iterate all files from a directory
for file_name in os.listdir(folder):
    # Construct old file name
    source = folder + file_name

    # Adding the count to the new file name and extension
    destination = folder + "Aug_" + str(count) + ".png"

    # Renaming the file
    os.rename(source, destination)
    count += 1
print('All Files Renamed')

print('New Names are')
# verify the result
res = os.listdir(folder)
print(res)


#--------------------------------------AUGMENTED TO TRAINING SET--------------------------------------
source = 'J:/No_label_spectro/Augment/E'
dest = 'J:/New-Total-Aug-Set/Train/E'
files = os.listdir(source)
rec_on_dest= os.listdir(dest)
no_of_files = 1564-(len(rec_on_dest))
print(no_of_files)

for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)
