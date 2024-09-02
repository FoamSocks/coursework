# Team_HP_CS4321_Midterm

## Directory
- best_models: Contains model checkpoints from best runs for fixed feature and fine tuned models. Entire models were saved (https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model).
- models: Contains artifacts for intermediate model runs. Checkpoints for trials were saved locally and are not in the repository.
- plots: Contains raw output images and artifacts for plots and figures used in the paper.
- scripts: Contains hamming scripts used and slurm output text files for various trials.
- trainer: Contains majority of source code for the project.
    - callbacks.py: Contains callback functions for model checkpoints and csv logging
    - data_class.py: Class for loading the coastal dataset images from file.
    - models_fixed.py: Functions for creation of fixed-feature models.
    - models_tuned.py: Functions for creation of fine-tuned models.
    - params.py: Parameter parser for use with scripts.
    - plots.py: Generates and saves plots for PCA, t-SNE, and classification report.
    - raw_data_plotter.py: Generates and saves t-SNE plot for raw data.
    - task.py: Contains code to load/transform data, load/train models, and evaluate on test data.
- environment.yml: YAML file for conda environment used in this project.

## Pytorch Configuration on Hamming
1. Create conda environment and activate.
```bash
conda create -n pytorch python=3.11
conda activate pytorch
```
2.  Reference for latest CUDA support and version compatibility: https://pytorch.org/blog/deprecation-cuda-python-support/. Install python and cudatoolkit from conda-forge.
```bash
conda install -c conda-forge python=3.11
conda install -c conda-forge cudatoolkit=11.7.*
```
4. Install pytorch with instructions from website: https://pytorch.org/get-started/locally/. Ensure versions match and are compatible with step 2.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
5. Test CUDA availability with pytorch.
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.current_device())"
``` 
Should output True and integers representing available GPUs.
```bash
True 0
```
6. If fails, try forcing a reinstall of pytorch.
```bash
pip install pytorch --force-reinstall
```