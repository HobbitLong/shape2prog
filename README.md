## Learning to Infer and Execute 3D Shape Programs

This repo covers the implementation for this ICLR 2019 paper:

"Learning to Infer and Execute 3D Shape Programs" [Paper](https://openreview.net/forum?id=rylNH20qFQ), [Project Page](http://shape2prog.csail.mit.edu). 

![Teaser Image](http://shape2prog.csail.mit.edu/shape_files/teaser.jpg)

## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0.

1. Clone this repo with:
	```
	git clone https://github.com/HobbitLong/shape2prog.git
	cd shape2prog
	```

2. Optional: you can consider setting up a virtual environment. Such environment (e.g. `shapeEnv`) can be created by
	```
	virtualenv -p python3 ~/env/shapeEnv
	```
	where `~/env/` indicates the directory to install the environment, and you can modify it accordingly. Activate `shapeEnv` by running
	```
	source ~/env/shapeEnv/bin/activate
	```

3. Install packages:
	```
	pip3 install -r requirements.txt
	```

## Download

Download data and model, assume now you are under `shape2prog` folder:
```
./download.sh
```

## Testing

Testing with downloaded models and data.
- `--model`: the directory to the model. Default: point to the model after guided adaptation for chair.
- `--data`: the directory to the data file, which is in hdf5 format.
- `--save_path`: the directory to save the output results, including reconstructed shapes (in voxel format), programs, and rendered images.
- `--save_prog`: turn on the option to save programs to .txt files, which will be saved in the `programs` folder under the specified `save_path`.
- `--save_img`: turn on the option to save rendered 2d images, which will be saved in the `images` folder under the specified `save_path`.
- `--num_render`: number of shapes to be rendered, starting from the first one. Default: 10.

Example: testing <i>chair</i> with pretrained models:
```
CUDA_VISIBLE_DEVICES=0 python test.py --model ./model/program_generator_GA_chair.t7 --data ./data/chair_testing.h5 --save_path ./output/chair/ --save_prog --save_img
```

Several modifications have been implemented in this repo beyond the original paper, which leads to improved reconstruction IoU. Such modifications include:
- Slightly modified sampling process for synthetic data.
- Increase the LSTM dimension of program executor.
- Further tunned hyper-parameters.

We tabulate the comparion of IoU bwteen original paper and this repo as follows.

|          |Chair | Table | Bed  | Sofa  | Cabinet |  Bench  |
|----------|:----:|:---:|:---:|:---:|:---:|:---:|
|  **Paper** | .591 | .516  | .367  | .597  |  .478  | .418 |
| **This Repo** | .663 | .560  | .439  | .649  |  .598  | .461 |

## Training

This repo provides the code to train those downloaded models from scratch.

#### Train Program Generator

Synthesize data (4GB):
```
python synthesize_shapes.py
```

Train the program generator with synthesized tables and chairs
```
CUDA_VISIBLE_DEVICES=0 python train_program_generator.py
```
By default, the checkpoints will be saved to `./model/ckpts_program_generator` folder.

#### Train Program Executor

Synthesize data (30GB):
```
python synthesize_blocks.py
```

Train the program executor with synthesized part-based programs.
```
CUDA_VISIBLE_DEVICES=0 python train_program_executor.py
```
By default, the checkpoints will be saved to `./model/ckpts_program_executor` folder.

#### Guided Adaptation
With the guidance of the learned program executor, the program executor can be adapted to other unseen furniture classes (bed, cabinet, sofa, and bench), as well as further improve the original classes (table and chair), in an unsupervised manner. We adapt a separate model for each object class. For example, run GA on <i>sofa</i> by:
```
CUDA_VISIBLE_DEVICES=0 python train_guided_adaptation.py --cls sofa
```
By default, it will automatically load the program generator and executor models we obtained by last two steps. The model after <i>guided adaptation</i> will be saved to the direcotry `./model/ckpts_GA_sofa`.


## Process Your Own Data

If your data is in .obj format, then the `generate_voxels.m` file under `matlab` folder illustrates the data preprocessing step used in this repo. 

If your shapes have already been voxelized, you might need to permute the voxel dimension to align with the model. The `generate_voxels.m` file might also give you some hint.


## Citation

If you find this repo useful for your research, please consider citing the paper

```
@inproceedings{tian2018learning,
  title={Learning to Infer and Execute 3D Shape Programs},
  author={Yonglong Tian and Andrew Luo and Xingyuan Sun and Kevin Ellis and William T. Freeman and Joshua B. Tenenbaum and Jiajun Wu},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

For any questions, please contact Yonglong Tian (yonglong@mit.edu).

## Acknowledgement

Part of the model code is inspired by [ImageCaptioning.pytorch](http://github.com/ruotianluo/ImageCaptioning.pytorch). The visualization code under `visualization` folder is slightly modified from [3dgan-release](https://github.com/zck119/3dgan-release/tree/master/visualization/python).

