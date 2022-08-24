Dependencies
~~~~~~~~~~~~
- Python
- Torch - Framework used for model training
- Matplotlib - Used for plotting graphs
- pycocotols(coco_eval and coco_utils) - Used to evaluate the model using cocometrics
- transforms - Contains methods for image manipulation
- engine - Used for training and model validation using cocometrics
- utils - Auxiliary tool used in the dataloader to load images
~~~~~~~~~~~~

### How to Run(Windows)

```console
python -m venv env
env/scripts/activate.ps1
pip install -r requirements.txt
python train.py
```

- link do the Dataset: https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
- link do the Benchmark: https://www.kaggle.com/code/nageshsingh/-mask-and-social-distancing-detection-using-vgg19/notebook