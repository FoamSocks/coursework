# NPS Thesis

## Contents
- data_processing: Data processing scripts to process labeled packet data into input to models
    - data_read.py
    - data_processing.py
    - data_split.py
    
- models: Scripts for models, testing, and model visualizations
    - Models:
        - models.py
        - callbacks.py
        - train.py
        - concrete_dropout.py
        - student_teacher.py
    - Testing:
        - bayesian_inference.py
        - st_inference.py
    - Visualizations
        - saliency_map.py
        - hessian.py
        - integratedhessian.py
        - Jupyter Notebooks
            - Probabilistic Prediction Plots
            - ROC AUC
            - Saliency Maps and Hessians