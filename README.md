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

