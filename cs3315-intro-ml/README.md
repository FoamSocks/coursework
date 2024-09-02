# CS3315 Machine Learning Project

## Project Objectives
1. Supervised Learning
    - kNN Classifier (Coble)
    - Logistic Regression (Coble)
    - Random Forest Classifier (Huang)
    - Multi-Layer Perceptron (Huang)

3. Unsupervised Learning (Kyler)
    - K-means clustering
    - Isolation forest
    - Other methods

## TensorFlow Configuration (on Ubuntu Linux)

1. Install miniconda (or anaconda). https://conda.io/projects/conda/en/stable/user-guide/install/linux.html
2. Create virutal environment
```bash
conda create --name tf python=3.9
conda activate tf
```
3. Follow TensorFlow installation instructions. Install tf-nightly instead of current TensorFlow build (due to incompatibilities with currently available TensorRT package from NVIDIA, 8.5.x)
4. Add CUDA path to LD_LIBRARY_PATH environment variable
```bash
cd $CONDA_PREFIX/etc/conda/activate.d
```
$CONDA_PREFIX is the conda environment folder, e.g. /home/miniconda3/envs/tf/.

Check with:
```bash
echo $CONDA_PREFIX
```

In env_vars.sh add:
```bash
export LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.9/site-packages/tensorrt/
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/python3.9/site-packages/tensorrt
```
5. Install cuda-nvcc
```bash
conda install -c nvidia cuda-nvcc
```
6. Restart the environment.
```bash
conda deactivate
conda activate tf
```
7. Verify successful install with following test code:

    - Should return a Tensor object:
    ```bash
    python3 -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    ```
    - Should list available GPUs:
    ```bash
    python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
    ```