# Introduction
Idea of this project is to understand and learn to use and manipulate data in `tfrecord` format.

# Dataset
Describe the dataset used in the project. Include details such as the source of the dataset, its format, and any preprocessing steps performed.

The dataset used in this project is available as a pickle file (`appml-assignment1-dataset-v2.pkl`) and a TFRecord file (`correct.tfrecord`). The pickle file contains features (`X`) and labels (`y`). The TFRecord file is used for creating a TensorFlow dataset. Both files are downloaded from a GitHub repository using the provided URLs in the code.

# Model Architecture
The model architecture includes:
<ol>
<li>Input layers for 'tickers', 'weekday', 'month', and 'hour'.</li>
<li>An imputer layer to handle missing values.</li>
<li>A normalization layer to normalize the input features.</li>
<li>IntegerLookup layers for categorical features ('weekday', 'month', 'hour').</li>
<li>Embedding layers for the categorical features.</li>
<li>A restMod (sequential) model with dense layers and dropout.</li>
<li>The output layer with softmax activation for multi-class classification.</li>
</ol>
  
# Training the Model
The model was compiled using `sparse_categorical_entropy` loss and the `SGD` optimizer. Trained the model for 20 epochs using a batch size of 32. Evaluated the model on the validation set during training to monitor its performance.. The training process includes:

<ol>
<li>Splitting the dataset into training, validation, and testing datasets.</li>
<li>Using ModelCheckpoint and EarlyStopping callbacks to save the best model and prevent overfitting.</li>
<li>Training the model for a specified number of epochs.</li>
</ol>

# Dependencies
<ul>
<li>tensorflow</li>
<li>keras</li>
<li>numpy</li>
<li>matplotlib</li>
</ul>

# Results
The model has less than 100K learnable parameters and achieved an accuracy of 25.00% on the validation set.

<img src="https://github.com/SanthoshV14/FX-price-prediction/blob/main/img/accuracy-plot.png" width=500 />

# Conclusion
IPython notebook project presented here demonstrates the process of building and training a model for multi-class classification using TensorFlow. The project utilizes a dataset containing features related to financial data, such as stock market information.

# Author
Santhos Vadivel </br>
Email - ssansh3@gmail.com </br>
LinkedIn - https://www.linkedin.com/in/santhosh-vadivel-2141b8126/ </br>
