DIQM
====
DIQM is a deep-learning based reference-based metric trained on [HDR-VDP](http://hdrvdp.sourceforge.net/wiki/). The main advanatage of DIQM is to transfer
HDR-VDP into a CNN that can efficiently predicts scores.

![HDR-VDP](images/diqm.png?raw=true "HDR-VDP")


DEPENDENCIES:
==============

Requires the PyTorch library along with Image, NumPy, SciPy, Matplotlib, glob2, pandas, and scikit-learn.

As the first step, you need to follow the [instructions for installing PyTorch](http://pytorch.org/).

To install dependencies, please use the following command: 

```
pip3 install numpy, scipy, matplotlib, glob2, pandas, image, scikit-learn, opencv-python. 
```

or check the prerequisites.txt file.

HOW TO RUN IT:
==============
To run our metric on images (i.e., JPEG, PNG, EXR, HDR, and MAT files),
you need to launch the file ```diqm.py``` and set the ground truth (```-src```) and the image to be tested (```-dst```). Some examples:

Testing images after inverse tone mapping operators:

```
python3 diqm.py HDR_ITMO -src tests/image_original.hdr -dst tests/image_eil.hdr
```

WEIGHTS DOWNLOAD:
=================
Weights can be downloaded here:
<a href="http://www.banterle.com/francesco/projects/norvdpnetpp/norvdpnetpp_itmo.pth">ITMO</a>.

Note that these weights are meant to model ONLY determined distortions; please see reference to have a complete overview.


DO NOT:
=======

There are many people use DIQM in an appropriate way:

1) Please do not use weights_nor_itmo for SDR images;

2) Please do not use weights for different distortions.

DATASET PREPARATION:
====================
Coming soon.

TRAINING:
=========
Coming soon.


REFERENCE:
==========

If you use DIQM in your work, please cite it using this reference:

```
@article{Artusi+2020,
  author       = {Alessandro Artusi and Francesco Banterle and Fabio Carrara and Alejandro Moreo},
  title        = {Efficient Evaluation of Image Quality via Deep-Learning Approximation
                  of Perceptual Metrics},
  journal      = {{IEEE} Trans. Image Process.},
  volume       = {29},
  pages        = {1843--1855},
  year         = {2020},
  url          = {https://doi.org/10.1109/TIP.2019.2944079},
  doi          = {10.1109/TIP.2019.2944079},
}
```
