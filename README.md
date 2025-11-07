# SLM-alignment
This is the repository for the article on "A practical guide to the alignment of defocused spatial light modulators for fast diffractive neural networks" from IOPscience.

# Calibration Demonstration
The following gif is an example of the calibration procedure performed in this project.

On the left, the raw images acquired by the camera.

On the right, the post-processed images for an easier ellipse fitting.

![](https://github.com/TTimTT/SLM-alignment/blob/main/docs/demo.gif?raw=true)

# Third party libraries

This project also uses various third parties libraries:

- [cudacanvas](https://github.com/OutofAi/cudacanvas), for displaying CUDA torch tensors from python to OpenGL efficiently. The provided code in our repository is a fork with adaptations for our SLMs.
- [Op Torch](), pyTorch library to perform various optical operations.
