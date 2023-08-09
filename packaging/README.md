This document explains the necessary steps to run inference using the trained model to obtain the spinal cord and lesion segmentations. Note that the documentation assumes that the user has `conda` installed on their system. Instructions on installing `conda` can be found [here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). 

### Step 1: Instructions for creating a virtual environment

1. Create a conda environment with the following command:
```
conda create -n venv_nnunet python=3.9
```

2. Activate the environment with the following command:
```
conda activate venv_nnunet
```

3. Install the required packages with the following command:
```
pip install -r requirements.txt
```

### Step 2: Instructions for running inference using the trained model

We provide two methods to run inference on a trained model to obtain the segmentations.

1. **On Individual Images**: This accepts a single image or a list of images. Note that in the case of a list of images, each input image must be separated by a space. Run the following command from the terminal:

```bash
python run_inference.py --path-images /path/to/image1 /path/to/image2 --path-out /path/to/output --path-model /path/to/model
```

2. On a Dataset: This method performs the inference on all the images in the given dataset. Run the following command from the terminal:

```bash
python run_inference.py --path-dataset /path/to/test-dataset --path-out /path/to/output --path-model /path/to/model
```

> **Note**
> The inference scripts also supports inference on a GPU. To do so, simply add the flag `--use-gpu` at the end of the above commands. By default, the inference is run on the CPU. 
