import os
import glob
import numpy as np
from PIL import Image
import pandas as pd
import csv

split_data = {}
for file in sorted(glob.glob('/content/testTrainMulti_7030_splits/*1.txt')):
  with open(file) as f:
    for line in f:
      split_data[line.split(' ')[0]] = int(line.split(' ')[1])

location = 'PATH TO YOUR HMDB DATASET'
train_data, train_labels = [], []
test_data, test_labels = [], []
val_data, val_labels = [], []
classes = []
labels = {}
for foldername in sorted(os.listdir(location)):
  classes.append(foldername)
for i in range(len(classes)):
  tmp = np.zeros(len(classes), dtype = np.int32)
  tmp[i] = 1
  labels[classes[i]] = tmp

print(classes)
# print(labels)

f1 = open('YOUR_PATH_HERE/train.csv', 'w')
f2 = open('YOUR_PATH_HERE/test.csv', 'w')
f3 = open('YOUR_PATH_HERE/val.csv', 'w')

for video in classes[:10]:
    for videoname in sorted(glob.glob(location+video+'/*')):
        print(videoname.split('/')[-1])
        print(video)
        print('='*50)
        if split_data[videoname.split('/')[-1]] == 1 :
          # create the csv writer
          train_writer = csv.writer(f1)

          # write a row to the csv file
          train_writer.writerow([videoname + ' '+ str(classes.index(video))])
   
        elif split_data[videoname.split('/')[-1]] == 2 :
          test_writer = csv.writer(f2)
          test_writer.writerow([videoname + ' '+ str(classes.index(video))])
        
        elif split_data[videoname.split('/')[-1]] == 0 :
          val_writer = csv.writer(f3)
          val_writer.writerow([videoname + ' '+ str(classes.index(video))])

f1.close()
f2.close()
f3.close()