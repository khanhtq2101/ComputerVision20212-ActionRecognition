import numpy as np
from PIL import Image
import os
import glob
from .preprocess_data import *
from torch.utils.data import Dataset, DataLoader



def get_video(opt, frame_path, Total_frames, train):
    """
        Args:
            opt         : config options
            frame_path  : frames of video frames
            Total_frames: Number of frames in the video
        Returns:
            list(frames) : random clip (list of frames of length sample_duration) from a video for training/ validation
        """
    rng = np.random.default_rng()
    frames = []
    seg_length = Total_frames / opt.n_segments

    for seg_idx in range(opt.n_segments):
      if train != 0:
        frames_idx = rng.choice([k for k in range(int(seg_length*seg_idx), int(seg_length*(seg_idx+1)))], size = 1, replace = False)
      elif train == 0:
        frames_idx = rng.choice([k for k in range(int(seg_length*seg_idx), int(seg_length*(seg_idx+1)))], size = opt.test_times, replace = False)

      i = 0
      loop = 0
      
      if opt.modality == 'RGB':
        for idx in frames_idx: 
          try:
            im = Image.open(os.path.join(frame_path, '%05d.jpg'%(idx+1)))
            frames.append(im.copy())
            im.close()
          except:
            pass

      elif opt.modality == 'Flow':  
          for idx in frames_idx:
            try:
              im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(idx+1)))
              im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(idx+1)))
              frames.append(im_x.copy())
              frames.append(im_y.copy())
              im_x.close()
              im_y.close()
            except:
              pass
                  
      elif  opt.modality == 'RGB_Flow':
          for idx in frames_idx:
            try:
              im   = Image.open(os.path.join(frame_path, '%05d.jpg'%(idx+1)))
              im_x = Image.open(os.path.join(frame_path, 'TVL1jpg_x_%05d.jpg'%(idx+1)))
              im_y = Image.open(os.path.join(frame_path, 'TVL1jpg_y_%05d.jpg'%(idx+1)))
              frames.append(im.copy())
              frames.append(im_x.copy())
              frames.append(im_y.copy())
              im.close()
              im_x.close()
              im_y.close()
            except:
              pass

    return frames


class HMDB51_test(Dataset):
    """HMDB51 Dataset"""
    def __init__(self, train, opt, split=None):
        """
        Args:
            opt   : config options
            train : 0 for testing, 1 for training, 2 for validation 
            split : 1,2,3 
        Returns:
            (tensor(frames), class_id ): Shape of tensor C x T x H x W
        """
        self.train_val_test = train
        self.opt = opt
        
        self.lab_names = sorted(set(['_'.join(os.path.splitext(file)[0].split('_')[:-2])for file in os.listdir(opt.annotation_path)]))
        # Number of classes
        self.N = len(self.lab_names)
        assert self.N == opt.n_classes

        self.lab_names = dict(zip(self.lab_names, range(self.N)))   # Each label is mappped to a number

        # indexes for training/test set
        split_lab_filenames = sorted([file for file in os.listdir(opt.annotation_path) if file.strip('.txt')[-1] ==str(split)])
        self.data = []                                     # (filename , lab_id)
        
        for file in split_lab_filenames:
            class_id = '_'.join(os.path.splitext(file)[0].split('_')[:-2])
            f = open(os.path.join(opt.annotation_path, file), 'r')
            for line in f: 
                # If training data
                if train==1 and line.split(' ')[1] == '1':
                    frame_path = os.path.join(opt.frame_dir, class_id, line.split(' ')[0][:-4])
                    if opt.only_RGB and os.path.exists(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))
                    elif os.path.exists(frame_path) and "done" in os.listdir(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))

                # Elif validation data        
                elif train == 2 and line.split(' ')[1] == '2':
                    frame_path = os.path.join(opt.frame_dir, class_id, line.split(' ')[0][:-4])
                    if opt.only_RGB and os.path.exists(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))
                    elif os.path.exists(frame_path) and "done" in os.listdir(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))
                
                #Test data
                elif train == 0 and line.split(' ')[1] == '0':
                    frame_path = os.path.join(opt.frame_dir, class_id, line.split(' ')[0][:-4])
                    if opt.only_RGB and os.path.exists(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))
                    elif os.path.exists(frame_path) and "done" in os.listdir(frame_path):
                        self.data.append((line.split(' ')[0][:-4], class_id))

            f.close()

    def __len__(self):
        '''
        returns number of test/train set
        ''' 
        return len(self.data)

    def __getitem__(self, idx):
        video = self.data[idx]
        label_id = self.lab_names.get(video[1])
        frame_path = os.path.join(self.opt.frame_dir, video[1], video[0])

        if self.opt.only_RGB:
            Total_frames = len(glob.glob(glob.escape(frame_path) +  '/0*.jpg'))
        else:
            #Total_frames = len(glob.glob(glob.escape(frame_path) +  '/0*.jpg'))
            Total_frames = len(glob.glob(glob.escape(frame_path) +  '/TVL1jpg_y_*.jpg'))
      
        Total_frames = min(len(glob.glob(glob.escape(frame_path) +  '/TVL1jpg_y_*.jpg')), len(glob.glob(glob.escape(frame_path) +  '/0*.jpg')))

        clip = get_video(self.opt, frame_path, Total_frames, self.train_val_test)
        

        return((scale_crop(clip, self.train_val_test, self.opt), label_id))