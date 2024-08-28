# CFIRSTNET: Comprehensive Features for Static IR Drop Estimation with Neural Network

## Introduction



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
