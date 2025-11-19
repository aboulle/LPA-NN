# X-ray diffraction Line Profile Analysis with deep Neural Networks (LPA-NN)

Jupyter notebooks to train and test a convolutional neural network devoted to the determination of crystallite size and/or microstrain from XRD data.

- Size: train and test
- Size_strain: train and evaluate a CNN with size and microstrain
- Spinel: a CNN to determine size and microstrain from a spinel phase formed by solid-state reaction between MgO and Al2O3. Experimental data, recorded in situ at 1200°C at the BM01 beamline (ESRF).

The notebooks require a working [tensorflow](https://www.tensorflow.org) installation, with GPU support highly recommended. Numpy and matplotlib are also required.

The datasets to train and test the notebooks are too large to be hosted on Github. They can be downloaded from the French research data repository: [https://doi.org/10.57745/SVQART](https://doi.org/10.57745/SVQART)
