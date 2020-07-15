# LT2212 V20 Assignment 3

## Part 1
Run a3_features.py by going to the folder where the a3_features.py is located via the terminal and enter the following:<br> 
`python3 a3_features.py <"input_file_name"> <csv_output_file> <dimensions_int> --test <test_percentage_as_int>` 

## Part 2

I did not declare the Sigmoid activation function excplicitly in the __init__ section of the model class since the loss function BCEWithLogitsLoss() (declared as variable "criterion" in def train_test_model) is supposed to combine a Sigmoid layer and the BCELoss in one single class (see Pytorch documentation for more information). The standardization of the input features with Scalar gives rise to some Datconversion warnings when the program is run in the mltgpu, but it seems to run nonetheless. 

## Part 3
I augmented the model by adding options for running the samples via the **rectifier**(--nonlin "relu") or **tanh** (--nonlin "tanh") activation function. The basic model, run on an input csv of 300 features and 80/20 train and test data split with 1500 samples for train and 500 for test, yields an accuracy of around 62-64 %. Unfortunately it doesn't seem to get much better than that, even when trying it through the other nonlinear functions and with different hidden layer sizes. It seems like there are so many parameters at play here (sample size, batch size, hidden layer size, learning ratio, feature table size, etc.) that it becomes difficult to pinpoint one single cause for the poor results.  

To run the model, please enter the following into the terminal opened in the folder of the a3_model.py file:<br> 
`python3 a3_model.py <input_file_name.csv> --sample <sample_size_as_int> --hidden <hidden_layer_size_as_int> --nonlin <"relu" or "tanh">` 

Sample size is optional, just like hidden size and nonlinear function, and if not specified then it will default to 1500 samples for training data and on third of that for the test data. 

## Part bonus
Go to the directory where a3_model_variant.py is located and run the following:<br>
`python3 a3_model_variant.py <input_file_name.csv> --sample <sample_size_as_int> --hidden_range <hidden_layer_range_as_int> --nonlin <"relu" or "tanh">` 

The terminal will once again show a bunch of DataConversionWarnings (sorry!), but should still run and produce a png file in the current directory. 

