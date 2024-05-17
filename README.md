# IID_SfmLearner
This is the official PyTorch implementation for training and testing depth estimation models using the method described in
> [**Image Intrinsic-Based Unsupervised Monocular Depth Estimation in Endoscopy**](https://ieeexplore.ieee.org/document/10530343)

> Bojian Li, Bo Liu, Miao Zhu, Xiaoyan Luo and Fugen Zhou

overview

## 📄 Citation
If you find our work useful in your research please consider citing our paper:
```
@article{Li2024Image,
  author={Li, Bojian and Liu, Bo and Zhu, Miao and Luo, Xiaoyan and Zhou, Fugen},
  journal={IEEE Journal of Biomedical and Health Informatics}, 
  title={Image Intrinsic-Based Unsupervised Monocular Depth Estimation in Endoscopy}, 
  year={2024},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/JBHI.2024.3400804}}
```

## ⚙️ Setup
We ran our experiments with PyTorch 1.11.0, CUDA 11.2, Python 3.8.13 and Ubuntu 18.04.

## 💾 Datasets
You can download the [Endovis or SCARED dataset](https://endovissub2019-scared.grand-challenge.org/) by signing the challenge rules and emailing them to [max.allan@intusurg.com](mailto:max.allan@intusurg.com),  you can download the Hamlyn dataset from this [website](http://hamlyn.doc.ic.ac.uk/vision/).

**Endovis split**
The train/test/validation split for Endovis dataset used in our works is defined in the  `splits/endovis`  folder.

**Data structure**
The directory of dataset structure is shown as follows:
```
/path/to/endovis_data/
  dataset1/
    keyframe1/
      left_img/
          000001.png
```
## 🖼️ Prediction for a single image
You can predict scaled disparity for a single image or a folder of images with:
```
python test_simple.py --image_path <your_image_or_folder_path> --model_path <your_model_path> --output_path <path to save results>
```

## ⏳ Training
You can train a model by running the following command:
```
python train.py --data_path <your_data_path> --log_dir <path_to_save_model>
```
## 📊 Evaluation
To prepare the ground truth depth maps run:
```
python export_gt_depth.py --data_path <your_data_path> --split <your_dataset_type>
```
You can evaluate a model by running the following command:
```
python evaluate_depth.py --data_path <your_data_path> --load_weights_folder <your_model_path> --eval_split <your_dataset_type>
```

## ✏️Acknowledgement
Our code is based on the implementation of [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner). We thank these authors for their excellent work and repository.
