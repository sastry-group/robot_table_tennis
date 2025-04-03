# robot_table_tennis

## Installing the Conda environments

Make sure you have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed. Then run:

```bash
conda env create -f RallyClipper/env.yml
conda env create -f BallTracker/env.yml
conda env create -f TableTracker/env.yml
```

## Setting up the Human Pose Tracker

For the following steps navigate to the HumanPoseTracker folder in this repo.
```bash
cd HumanPoseTracker
```

### Human Pose Tracker environment setup:

```bash
conda env create --file env.yml
conda activate pose
conda remove cuda-version
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -n pose gxx_linux-64
```

### PHALP setup:

Run:
```bash
pip install git+https://github.com/brjathu/PHALP.git
```
Then replace phalp/trackers/PHALP.py with the local version in this repo. To help find the file, open a python shell and run:
```bash
import phalp
phalp.__file__
```

### SMPL Mesh

Make sure to find you have the `phalp/3D/models/smpl/SMPL_NEUTRAL.pkl` files


## Downloading Ball Tracking Model
For the following steps navigate to the HumanPoseTracker folder in this repo.
```bash
cd BallTracker
```

Once there run the following commands.
```bash
conda deactivate
conda activate ball
pip install gdown
rm -r finetune
rm -r ckpts
gdown 1b7esQo0NNkFutR5ScC1KKWW0zyUGjZ1E
gdown 1sK9H5_5kbHegb-_b-5PuDeifXNQQeMHv
unzip finetune.zip
unzip ckpts.zip
rm finetune.zip
rm ckpts.zip
conda deactivate
```


## Test 

To test the repository run the following. 

```bash
chmod +x run_pipeline.sh
./run_pipeline.sh --gpu_ids "0 1 2 3 4 5" all
```

You can replace "0 1 2 3 4 5" with the ids of the gpus on your machine.