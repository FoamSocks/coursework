# Tensorflow Configuration

1. Log into hamming. 
2. From a submit node, request a GPU compute node (replace partition with whichever you are using, e.g. beards, barton, monaco, etc). Specify memory amount (e.g. 32G or 32000) and time (days-hours:minutes:seconds)
```bash
srun --pty --mem=XXG –partition=beards -gres=gpu:1 --time dd-hh:mm:ss bash
```
3.	Test GPU access. If no output or nvidia-smi is not found, verify you are on a partition with access to GPUs.
```bash
nvidia-smi
```
Output looks like this:
```
Wed Mar 15 11:29:43 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A100-SXM...  On   | 00000000:01:00.0 Off |                    0 |
| N/A   24C    P0    53W / 400W |      0MiB / 40960MiB |      0%      Default |
|                               |                      |             Disabled |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```
4. Load conda module. If specific version below is not found, type 'module avail' to list available miniconda versions and choose one.
```bash
module load lang/miniconda3/4.5.12 
```
Prompt should now be prepended with (env). Example below:
```bash
(base) [first.last@compute-9-40 ~]  # prompt shows we are in the base environment
```
5. Create conda environment. Choose any name for \<name\>. 
```bash
conda create --name <name> python=3.9   # replace <name> with env name
```
Going forward we will use 'tf' as an example. 
```bash
conda create --name tf python=3.9
```
This creates a conda enviornment named 'tf.' Activate and deactive the environment with:
```bash
conda activate tf   # to activate environment
conda deactivate    # to return to base environment
```
You should see the prompt change to reflect the new environment.
```bash
(tf) [first.last@compute-9-40 ~]
```
6. Install cudnn and cudatoolkit
```bash
conda install -c conda-forge cudatoolkit=11.6 cudnn=8.1.0
```
7. Link CUDA libraries to $LD_LIBRARY_PATH. env_vars.sh will run when the environment is activated and link the libary directory to $LD_LIBRARY_PATH.
```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
8. Update pip.
```bash
pip install --upgrade pip
```
9. Install Tensorflow.
```bash
pip install tensorflow
```
10. Install TensorRT.
```bash
pip install tensorrt
```
11. Add TensorRT libary director to $LD_LIBRARY_PATH.
```bash
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/python3.9/site-packages/tensorrt/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```
$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh should now contain two export lines, one ending in lib (step 7 above) and one ending in tensorrt.

12. Deactivate and activate the environment to run the commands in env_vars.sh. Commands will be run every time the environment is activated from now on, and the libraries will be properly linked each time.

```bash
conda deactivate
conda activate tf
```

13. Tensorflow expects TensorRT version 7 for libnvinfer_plugin and libnvinfer. Compiled packages on all repositories (including pypi) will be version 8. To avoid compiling TensorRT 7.x from source, add symlinks for version 7 linking to version 8.
```bash
cd $CONDA_PREFIX/lib/python3.9/site-packages/tensorrt/
ln -s libnvinfer_plugin.so.8 libnvinfer_plugin.so.7
ln -s libnvinfer.so.8 libnvinfer.so.7
```
14. Test installation
```bash
python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
Should only return information messages (preceded by 'I'), and then a final line with the GPU device information. Output looks like this:
```
2023-03-15 11:30:55.729533: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
Verify Tensors can be properly loaded to the GPU and opertions run correctly.
```
python3 -c "import tensorflow as tf; x=tf.Variable(2.0); print(tf.square(x))"
```
Output looks like this:
```
2023-03-15 11:35:31.273276: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1613] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 38220 MB memory:  -> device: 0, name: NVIDIA A100-SXM4-40GB, pci bus id: 0000:01:00.0, compute capability: 8.0
tf.Tensor(4.0, shape=(), dtype=float32)
```