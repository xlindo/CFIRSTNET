# CFIRSTNET: Comprehensive Features for Static IR Drop Estimation with Neural Network

## Introduction

We propose a comprehensive solution to combine both the advantages of image-based and netlist-based features in neural network framework and obtain high-quality IR drop prediction very effectively in modern designs. A customized convolutional neural network (CNN) is developed to extract PDN features and make static IR drop estimations. Trained and evaluated with the open-source dataset, experiment results show that we have obtained the best quality in the benchmark on the problem of IR drop estimation in ICCAD CAD Contest 2023.

## Install & Build

Our codes are seperated into two parts, data preprocess part coding with C++, and the other part including model coding with Python. To run these codes, you must follow the following instructions.

1. Install Conda (Anaconda is tested) and CMake.

2. Create a conda environment with Python 3.10 and activate it.

    ```Command-line
    conda create -n myenv python=3.10 -y
    conda activate myenv
    ```

3. Install requirements for Conda environment.

    ```Command-line
    conda install --file conda_requirements.txt -y
    ```

4. Build the C++ codes for data preprocess with CMake.

    ```Command-line
    cmake -S ./c++ -B ./c++/build
    cmake --build ./c++/build
    cp ./c++/build/libdata.so ./src/libdata.so
    ```

5. Install requirement for Python environment.

    ```Command-line
    pip install -r py_requirements.txt
    ```

6. Run Training

    ```Command-line
    python train.py
    ```
