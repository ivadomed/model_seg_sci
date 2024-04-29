# SCISeg: Automatic Segmentation of T2-weighted Intramedullary Lesions in Spinal Cord Injury

[![medRxiv](https://img.shields.io/badge/medRxiv-10.1101/2024.01.03.24300794v2-blue.svg)](https://www.medrxiv.org/content/10.1101/2024.01.03.24300794v2.full.pdf)

This repository contains the code for deep learning-based segmentation of the spinal cord and hyperintense lesions in spinal cord injury (SCI). The code is based on the [nnUNetv2 framework](https://github.com/MIC-DKFZ/nnUNet).


## Model Overview

The model was trained on raw T2-weighted images of SCI patients from multiple (three) sites. The data included images with both axial and sagittal resolutions. To ensure uniformity across sites, all images were initially re-oriented to RPI. Given an input image, the model is able to segment *both* the lesion and the spinal cord. The model also works well on degenerative cervical myelopathy (DCM) lesions. 

<img width="1000" alt="figure2_fixed" src="https://github.com/ivadomed/model_seg_sci/assets/53445351/e7492462-18aa-4f7d-a03e-22863efaff72">

## Using SCIseg

### Install dependencies

- [Spinal Cord Toolbox (SCT) v6.2](https://github.com/spinalcordtoolbox/spinalcordtoolbox/releases/tag/6.2) or higher -- follow the installation instructions [here](https://github.com/spinalcordtoolbox/spinalcordtoolbox?tab=readme-ov-file#installation)
- [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) 
- Python (v3.9)

Once the dependencies are installed, download the latest SCIseg model:

```bash
sct_deepseg -install-task seg_sc_lesion_t2w_sci
```

### Getting the lesion and spinal cord segmentation

To segment a single image, run the following command: 

```bash
sct_deepseg -i <INPUT> -task seg_sc_lesion_t2w_sci
```

For example:

```bash
sct_deepseg -i sub-001_T2w.nii.gz -task seg_sc_lesion_t2w_sci
```

The outputs will be saved in the same directory as the input image, with the suffix `_lesion_seg.nii.gz` for the lesion 
and `_sc_seg.nii.gz` for the spinal cord.

## Citation Info

If you find this work and/or code useful for your research, please cite our paper:

```
@article {Naga Karthik2024.01.03.24300794,
	author = {Enamundram Naga Karthik* and Jan Valosek* and Andrew C. Smith and Dario Pfyffer and Simon Schading-Sassenhausen and Lynn Farner and Kenneth A. Weber II and Patrick Freund and Julien Cohen-Adad},
	title = {SCIseg: Automatic Segmentation of T2-weighted Intramedullary Lesions in Spinal Cord Injury},
	elocation-id = {2024.01.03.24300794},
	year = {2024},
	doi = {10.1101/2024.01.03.24300794},
	publisher = {Cold Spring Harbor Laboratory Press},
	URL = {https://www.medrxiv.org/content/early/2024/04/21/2024.01.03.24300794},
	eprint = {https://www.medrxiv.org/content/early/2024/04/21/2024.01.03.24300794.full.pdf},
	journal = {medRxiv},
	note = {*Shared first authorship}
}

```
