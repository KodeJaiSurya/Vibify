##  Hyperparameter Tuning

In this code (emotion_hyperparameter_tuning.py), hyperparameter tuning is conducted using Keras Tuner's RandomSearch method, which randomly samples from a defined search space to identify the best combination of hyperparameters for optimizing validation accuracy. The search space includes key hyperparameters such as the number of convolutional layers (conv_layers), the number of filters per layer (filters), the number of units in the dense layer (units), and the learning rate (learning_rate). The model is designed for a multi-class emotion classification task with input data shaped as 48x48 grayscale images. Each trial evaluates a unique set of hyperparameters, and the results are logged using MLFlow for analysis and reproducibility.

### Summary Chart of Hyperparameter Tuning

| Hyperparameter             | Search Space                             | Description                                                                 |
|----------------------------|------------------------------------------|-----------------------------------------------------------------------------|
| **conv_layers**             | Integer: [1, 3]                          | Number of convolutional layers to include in the model.                    |
| **filters (per layer)**     | Integer: [32, 128], step=32              | Number of filters for each convolutional layer.                            |
| **units (dense layer)**     | Integer: [64, 512], step=64              | Number of units in the fully connected dense layer.                        |
| **learning_rate**           | Float: [1e-5, 1e-2], log-sampling        | Learning rate for the Adam optimizer, selected using logarithmic scaling.  |
| **Features Attempted**      | Convolutional layers, dense units, learning rate | Combination of architectural and optimization parameters.         |

##  Experiment Tracking and Results

MLFlow is used to track the training and tuning of the model. It helps log important information like hyperparameters, model performance metrics, and the best model version. The main performance metric, validation accuracy, is recorded to see how well each combination of hyperparameters performs. After finding the best model, it is saved using mlflow.keras.log_model(), making it easy to load and use later. This process helps keep track of all the details so that the model can be recreated and compared later.

### Pictures of the MLFlow Implementation
![alt text](image.png)
![alt text](image-1.png)
![alt text](image-3.png)

## Model Sensitivity Analysis

Based on the hyperparameter tuning results obtained so far, we can observe how different parameter settings are affecting the model's performance (validation accuracy). While we have not completed all the iterations yet, we can analyze the trends in the trials performed so far:

1) Number of Convolutional Layers (conv_layers): All trials so far used 1 convolutional layer, and this configuration appears to provide a decent starting point. However, as the iterations are still ongoing, we expect that testing with more convolutional layers might improve performance, and further trials could explore the effects of adding layers.

2) Number of Filters (filters_0):The number of filters varied between 32, 64, and 96. From the current results, 64 filters showed the best performance, resulting in a validation accuracy of 0.5748, while 96 filters performed worse with 0.4975 accuracy. This suggests that using a moderate number of filters may be more effective than having too many or too few.

3) Learning Rate (learning_rate): Learning rate values ranged from 0.000086 to 0.00158. The learning rate of 0.000862 achieved the best validation accuracy of 0.5748, indicating a good balance for convergence. 

4) Number of Units in Dense Layer (units):The number of units varied from 64 to 384. The trial with 384 units achieved the best result, with a validation accuracy of 0.5748, suggesting that a larger number of units can improve the model's ability to capture complex patterns. 

These results represent the findings from the hyperparameter tuning process so far. We are still in the process of running additional iterations, as each one takes around an hour to complete. In addition to testing various hyperparameters, we are also experimenting with modifying the number of epochs, number of trials, and executions per trial to further optimize the model. 