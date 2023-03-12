# Quality Inspection Platform

## Model training

The `model_training_experiment.ipynb` notebook contains code for experimenting with image classification using a pre-trained Neural Network from Keras Applications with pre-trained weights.

This code can be used for the following:

1. Downloading image datasets from a certain URL
2. Visualizing part of the dataset
3. Retraining a pre-made Neural Network from Keras Applications with pre-trained weights
4. Graphing metrics during training
5. Calculating and plotting the confusion matrix
6. Saving the trained model
7. Saving an optimized model

### How to Use

To use this code, simply run the Jupyter notebook and follow the instructions in the comments. The code will download the image dataset, visualize part of the dataset, retrain a pre-made Neural Network with pre-trained weights, graph metrics during training, calculate and plot the confusion matrix, and save the trained and optimized models.

```bash
jupyter nbconvert --to script model_training_experiment.ipynb
```

### Disclaimer

This notebook is not meant to be used for production purposes. It is meant for experimentation purposes only.
