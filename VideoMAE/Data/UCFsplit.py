import os
import glob
label = {}
with open('YOUR_PATH_HERE/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/classInd.txt')  as f:
  for line in f:
    label[line.split(' ')[1][:-1]] = int(line.split(' ')[0]) - 1
labels_10 = sorted(os.listdir('/content/drive/MyDrive/UCF-101'))[:10]

import csv
location = 'PATH_TO_UCF101_DATASET'
f1 = open('YOUR_PATH_HERE/Data/ucf10/train.csv', 'w')
f2 = open('YOUR_PATH_HERE/Data/ucf10/test.csv', 'w')
f3 = open('YOUR_PATH_HERE/Data/ucf10/val.csv', 'w')
train_writer = csv.writer(f1)
test_writer = csv.writer(f2)
val_writer = csv.writer(f3)

# for file in glob.glob('/content/drive/MyDrive/Project_2/Data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/*.txt'):
#     if file.split('/')[-1] == 'classInd.txt':
#       continue
#     if 'trainlist' in file.split('/')[-1]:
train_dir = 'YOUR_PATH_HERE/Data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/trainlist01.txt'
test_dir = 'YOUR_PATH_HERE/Data/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist/testlist01.txt'
with open(train_dir) as f:
  for line in f: 
    # print(line)
    if line.split('/')[0] in labels_10:
      # print('line split', line[:-2].split(' '))
      row = line[:-1].split(' ')
      train_writer.writerow([location + row[0] + ' ' + str(int(row[1]) - 1)])

    # elif 'testlist' in file.split('/')[-1]:
with open(test_dir) as f:
  for line in f: 
    # print(line)
    if line.split('/')[0] in labels_10:
      # print('line split', line.split(' '))
      # print('row = ', location + line[:-1] + ' ' + str(label[line.split('/')[0]]))
      test_writer.writerow([location + line[:-1] + ' ' + str(label[line.split('/')[0]])])
      val_writer.writerow([location + line[:-1] + ' ' + str(label[line.split('/')[0]])])
# data = split_f.readlines()
# print('data = ', data)
# for line in data:
#     line_info = line.split(' ')
#     print('line_info =', line_info)
f1.close()
f2.close()
f3.close()