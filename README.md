# **ComputerVision20212-ActionRecognition**

Human action recognition (HAR) is an important task in the computer vision domain, especially in many situations such as video surveillance, video content analysis, video security control. However, it is a challenging task due to background clutter, lighting and the fact that human actions are usually variant over time, from different viewpoints, and occluded by other objects in environment
In this Captone Project, we focuse on studying action recognition with multi-modality. We examine and evaluate 3 different approaches on the two main datasets.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## Requirements

- Python 3.6 or higher
- [PyTorch](https://pytorch.org/) and [torchvision](https://github.com/pytorch/vision). <br>
- OpenCV with GPU support
- [timm==0.4.8/0.4.12](https://github.com/rwightman/pytorch-image-models)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
## Dataset  

* We use 2 datasets HMDB51 and UCF101 with extracted first 10 classes in each dataset. The full dataset and splits can be download from:

    [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

    [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)
* For Temporal Segments Network and Motion-Augmented RGB Stream, download dataset [here](https://drive.google.com/file/d/1bgruuIBdLm2uBU9cQUVpHLIemLRyy50a/view?usp=sharing) and annotation [file](https://drive.google.com/file/d/1bgruuIBdLm2uBU9cQUVpHLIemLRyy50a/view?usp=sharing).

## Methology

Three model are implemented and evaluated on above datasets, which are Temporal Segment Network (TSN), Motion-Augmented RGB Stream (MARS), Video Masked AutoEncoders (VideoMAE). Detail implementation, instruction for training and our evaluation result are included in each corresponding folder.
1. [Temporal Segment Network](TSN)
2. [Motion-Augmented RGB Stream](MARS)
3. [Video Masked AutoEncoders](VideoMAE)