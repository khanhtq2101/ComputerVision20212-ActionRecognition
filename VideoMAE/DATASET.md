# Data Preparation

We have pre-trained and fine-tuned our VideoMAE on [UCF101](https://www.crcv.ucf.edu/data/UCF101.php) and [HMDB51](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/) with only extracted first 10 classes for each dataset.

- The pre-processing of **HMDB10** can be summarized as following:

  1. Download the dataset from [official website](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads).
  2. Run the file [HMDBsplit.py](Data/HMDBsplit.py) to generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations). The annotation usually includes `train.csv`, `val.csv` and `test.csv`. The format of `*.csv` file is like:

     ```
     dataset_root/video_1.mp4  label_1
     dataset_root/video_2.mp4  label_2
     dataset_root/video_3.mp4  label_3
     ...
     dataset_root/video_N.mp4  label_N
     ```
- The pre-processing of **UCF10** can be summarized as following:

  1. Download the dataset from [official website](https://www.crcv.ucf.edu/data/UCF101.php).
  2. Run the file [UCFsplit](Data/UCFsplit.py) to generate annotations needed for dataloader ("<path_to_video> <video_class>" in annotations). The annotation usually includes `train.csv`, `val.csv` and `test.csv` ( here `test.csv` is the same as `val.csv`). The format of `*.csv` file is like:

     ```
     dataset_root/video_1.mp4  label_1
     dataset_root/video_2.mp4  label_2
     dataset_root/video_3.mp4  label_3
     ...
     dataset_root/video_N.mp4  label_N
     ```