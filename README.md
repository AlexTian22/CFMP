This repository provides the official PyTorch implementation for the paper: Fast and Accurate Visuomotor Imitation Learning via 2D Consistency Flow Matching Policy.
## 📦 Installation

Follow the steps below to set up the environment and install all required dependencies.

### 1️⃣ Create and Activate the Conda Environment
Use the provided `environment.yml` file to create the environment:
```bash
conda env create -f environment.yml
conda activate cfmp
pip install -e .
```
2️⃣ Environment Dependency Notes ⚠️
For all environments except Robomimic (PushT, BlockPush, Kitchen, Metaworld, LIBERO), use:
```bash
pip install robosuite==1.4.0
```
For Robomimic, use:

```bash
pip install robosuite==1.2.0
```
If you need both, consider creating separate conda environments to avoid package conflicts.

### 3️⃣ Install Metaworld
This project depends on the Metaworld simulation environment:

```bash
cd Metaworld
pip install -e .
cd ..
```
The -e flag installs the package in editable mode, so changes to the source code will take effect immediately.

### 4️⃣ Install LIBERO
We use the LIBERO dataset and API:

```bash
git clone git@github.com:Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e .
cd ..
```
✅ At this point, the environment and dependencies are ready. 

## 📁 Dataset Preparation

Follow the steps below to download and organize the datasets required for this project.

---

### 1️⃣ State-Based Tasks (Low-Dimensional Observations)
These datasets are sourced from the original [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) project.

| Task          | Description                                           | Download Link |
|---------------|-------------------------------------------------------|---------------|
| **Push-T**    | Push environment from Transporter benchmark           | [pusht.zip](https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip) |
| **Lift, Can** | Robomimic low-dimensional benchmark tasks              | [robomimic_lowdim.zip](https://diffusion-policy.cs.columbia.edu/data/training/robomimic_lowdim.zip) |
| **BlockPush** | Block pushing task from DMControl                      | [block_pushing.zip](https://diffusion-policy.cs.columbia.edu/data/training/block_pushing.zip) |
| **Kitchen**   | Kitchen manipulation tasks from FrankaKitchen          | [kitchen.zip](https://diffusion-policy.cs.columbia.edu/data/training/kitchen.zip) |

After downloading and unzipping, place the folders in:


---

### 2️⃣ Image-Based Tasks
We collect image-based datasets for **MetaWorld** and **LIBERO** environments.

| Environment  | Description                               | Download |
|--------------|-------------------------------------------|----------|
| **MetaWorld**| High-resolution image-based manipulation tasks | *https://huggingface.co/datasets/ShuaiTian/MetaWorld_Expert_10_Demos* |
| **LIBERO**   | General-purpose robot manipulation tasks  | *https://huggingface.co/datasets/ShuaiTian/LIBERO_6Tasks_Expert_50_Demos* |

Once released, you can download them via:
```bash
git lfs install
git clone https://huggingface.co/datasets/ShuaiTian/MetaWorld_Expert_10_Demos
git clone https://huggingface.co/datasets/ShuaiTian/LIBERO_6Tasks_Expert_50_Demos
```
3️⃣ LIBERO Dataset Placement
After downloading the LIBERO dataset, place it in:

swift
```bash
/opt/ts/cfmp/data/libero/libero_robomimic_format_demo50
```
Also copy the initialization files:

```bash
cp -r LIBERO/libero/libero/init_files /opt/ts/cfmp/data/libero
```
✅ At this point, all datasets are prepared and ready for Training and Inference.


## ⚙️ Simulation Training and Evaluation Instructions

### 1️⃣ Train
This section provides commands to train models on various tasks and methods in this project. Below are the training commands for different simulation environments and methods.

* Push-T — DP
```bash
python train.py --config-dir=. --config-name=state_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:2 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```
* Block-Push — DP
```bash
python train.py --config-dir=. --config-name=state_blockpush_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:3 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```
* Kitchen — DP
```bash
python train.py --config-dir=. --config-name=state_kitchen_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:3 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```
* Robomimic —DP
```bash
python train.py --config-dir=. --config-name=state_liftph_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:3 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```
* MetaWorld — RFMP (Example: coffee-push)
```bash
CUDA_VISIBLE_DEVICES=2 python train.py --config-dir=diffusion_policy/config --config-name=mw_fm.yaml training.device=cuda:0 training.seed=1 taskn="coffee-push"
```
* LIBERO — CFMP (Example: KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo)
```bash
CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 python train.py --config-name=libero_cfm task.task_name=libero_90 task=libero_image_abs task.dataset_type="KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo" training.device=cuda:0 taskn="KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo"
```

### 2️⃣ Evaluation

To evaluate a trained model, simply specify the checkpoint file and the output directory. The evaluation script will produce the success rate and inference videos. 

Here we use the LIBERO environment task **KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo** as an example.

```bash
python eval.py --checkpoint data/outputs_libero_dp/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo/checkpoints/epoch=0000-test_mean_score=0.980.ckpt -o data/KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo_eval_output
```


## 🏷️ License
This repository is released under the MIT license. See [LICENSE](LICENSE) for additional details.


## 🙏 Acknowledgement
We would like to thank the authors and developers of the following projects, which inspired or supported our work:

- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [3D Diffusion Policy](https://github.com/YanjieZe/3D-Diffusion-Policy)
- [AdaFlow](https://github.com/hxixixh/AdaFlow)
- [FlowPolicy](https://github.com/zql-kk/FlowPolicy)