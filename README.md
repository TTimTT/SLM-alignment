# Description
This is the repository for the article on "A practical guide to the alignment of defocused spatial light modulators for fast diffractive neural networks" from IOPscience.

# Calibration Demonstration
The following gif is an example of the calibration procedure performed in this project.

On the left, the raw images acquired by the camera.

On the right, the post-processed images for an easier ellipse fitting.

![](https://github.com/TTimTT/SLM-alignment/blob/main/docs/demo.gif?raw=true)

# Requirements
This project was developped using the following required system components:

- SLMs model
- Cameras model
- Linux + python version or conda
- Nvidia graphic card

# Installation
The first step is to install the provided conda environment:

```
conda env create -f environment.yml
``` 

After the environment was created you can activate it and install the third parties libraries:
```
conda activate slm-alignment
```

Installing our modified version of cudacanvas:
```
cd library/thirdparty/cudacanvas
pip install .
```


# Example

We provide a notebook `example/calibration.ipynb` to illustrate how the screens and camera are instantiated. Afterwards, a simple function executes the calibration procedure which displays the final detected ROIs.

# Third party libraries

This project uses various third parties libraries:

- [cudacanvas](https://github.com/OutofAi/cudacanvas), for displaying CUDA torch tensors from python to OpenGL efficiently. The provided code in our repository is a fork with adaptations for our SLMs.
- [Op Torch](), pyTorch library to perform various optical operations.

# Citing our work
