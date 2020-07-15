import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import csv
from sklearn.utils import shuffle
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt



def split_df(df):
    train_df = df[df["Type"] == "train"]
    test_df = df[df["Type"] == "test"]
    test_df.reset_index(inplace=True, drop=True)  
    return train_df, test_df

def extract_two_random_docs(df):
    two_random_rows_in_df = df.sample(n=2)  
    doc1 = two_random_rows_in_df.iloc[0].to_numpy()    
    doc2 = two_random_rows_in_df.iloc[1].to_numpy()

    return doc1, doc2
    
def make_sample(doc1, doc2):    
    author1 = doc1[0]
    author2 = doc2[0]
    doc1_vector = doc1[2:]
    doc2_vector = doc2[2:]
    
    if author1 == author2:
        sample_with_same_author = (doc1_vector + doc2_vector, 1)
        return sample_with_same_author
    else:
        sample_with_different_authors = (doc1_vector + doc2_vector, 0)
        return sample_with_different_authors
    
def get_samples(df, sample_size):
    limit_size_per_sample_type = sample_size/2
    samples = []
    same_author_sample_count = 0
    different_author_sample_count = 0
    
    while len(samples) < sample_size:
        random_doc1, random_doc2 = extract_two_random_docs(df)
        sample = make_sample(random_doc1, random_doc2)
        if sample[1] == 1 and same_author_sample_count < limit_size_per_sample_type:
            samples.append(sample)
            same_author_sample_count += 1
        elif sample[1] == 0 and different_author_sample_count < limit_size_per_sample_type:
            samples.append(sample)
            different_author_sample_count += 1
    return shuffle(samples)

#Two custom dataset classes to allow for reading and processing the data:
class Train_Data(Dataset):                
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__(self):
        return len(self.X_data)

class Test_Data(Dataset):                  
    def __init__(self, X_data):
        self.X_data = X_data
    def __getitem__(self, index):
        return self.X_data[index]
    def __len__(self):
        return len(self.X_data)

class Neuralnette (nn.Module):
    
    def __init__(self, input_size, hidden_size, nonlinear_fun):
        super(Neuralnette, self).__init__()
        self.nonlinear_fun = nonlinear_fun
        if hidden_size is None:
            self.input_layer = nn.Linear(input_size, 42) #the output size 42 as default value was chosen at random
            self.output_layer = nn.Linear(42, 1)
        else:
            self.input_layer = nn.Linear(input_size, hidden_size)
            self.hidden_layer = nn.Linear(hidden_size, hidden_size)   
            self.output_layer = nn.Linear(hidden_size, 1)             
            
        if self.nonlinear_fun == "relu":    
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=0.1) 
            self.batchnorm1 = nn.BatchNorm1d(hidden_size)
            self.batchnorm2 = nn.BatchNorm1d(hidden_size)
            
        elif self.nonlinear_fun == "tanh":
            self.tanh = nn.Tanh()
            self.dropout = nn.Dropout(p=0.1) 
            self.batchnorm1 = nn.BatchNorm1d(hidden_size)
            self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, inputs):
        x = self.input_layer(inputs)
        if self.nonlinear_fun == "relu":
            x = self.relu(self.input_layer(inputs))
            x = self.batchnorm1(x)
            x = self.relu(self.hidden_layer(x))
            x = self.batchnorm2(x)
            x = self.dropout(x)
        elif self.nonlinear_fun == "tanh":
            x = self.tanh(self.input_layer(inputs))
            x = self.batchnorm1(x)
            x = self.tanh(self.hidden_layer(x))
            x = self.batchnorm2(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        
        return x
    
    
#The only thing you need to ensure is that number of output features of one layer should be equal 
#to the input features of the next layer.
 

def calculate_accuracy(y_pred, y_test): #we take the predicted and actual output as the input.
    #The predicted value y_pred is rounded off to convert it into either a 0 or a 1:
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    
    return acc

def train_test_model(featurefile, sample_size, hidden_size, nonlinear_fun):
    
    #reading the csv file containing the feature table:
    df = pd.read_csv(featurefile)
    
    #split the dataframe into two new ones: one with the 'test'-marked docs, and one with the 'train'-marked
    train, test = split_df(df) 
    
    #generate samples out of the feature tables: 
    sample_size_train = sample_size
    sample_size_test = sample_size/3
    train_samples = get_samples(train, sample_size_train)
    test_samples = get_samples(test, sample_size_test)

    #extract document vectors and corresponding labels into four different sets: 
    X_train = np.array([sample_vector[0] for sample_vector in train_samples])
    X_test = np.array([sample_vector[0] for sample_vector in test_samples]) 
    y_train = np.array([sample_label[1] for sample_label in train_samples])
    y_test = np.array([sample_label[1] for sample_label in test_samples])
        
   
    #Standardizing features by removing the mean and scaling to unit variance:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    #epoch = when an entire dataset is passed forward and backward through the neural network.
    epochs = 50
    #batch size = total number of training examples present in a single batch.
    batch_size = 64
    #learning rate = specifies by how much we should change our model parameters inbetween epochs:
    learning_rate = 0.001
    
    train_data = Train_Data(torch.FloatTensor(X_train),
                        torch.FloatTensor(y_train))
    test_data = Test_Data(torch.FloatTensor(X_test))
    
    train_loader  = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=1)
 
    #check if GPU is active, otherwise use the CPU:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    #initializing the model:
    input_size = X_train.shape[1]
    model = Neuralnette(input_size, hidden_size, nonlinear_fun)
    model.to(device)
    
    #initialize optimizer and choice of loss function:
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

    #model.train() tells PyTorch that you’re in training mode: 
    model.train()    
    for e in range(1, epochs+1):  
        epochs_loss = 0
        epochs_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
        
            y_pred = model(X_batch)
        
            loss = criterion(y_pred, y_batch.unsqueeze(1))
            acc = calculate_accuracy(y_pred, y_batch.unsqueeze(1))
        
            loss.backward()
            optimizer.step()
            
            epochs_loss += loss.item()
            epochs_acc += acc.item()
            
        #print(f'Epoch {e+0:03}: | Loss: {epochs_loss/len(train_loader):.5f} | Acc: {epochs_acc/len(train_loader):.3f}')
       
   #intantiate a list that will hold the predictions:
    y_pred_list = []
    model.eval()
    with torch.no_grad():
        for X_batch in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            y_pred = torch.sigmoid(y_pred)
            y_pred_b = y_pred
            y_pred_tag = torch.round(y_pred)
            y_pred_list.append(y_pred_tag.cpu().numpy())
    
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]

    #precision, recall, F1-score and accuracy:
    #print(classification_report(y_test, y_pred_list))  y_test = y_true, y_pred_list = y_pred
    #print("Accuracy score: ", accuracy_score(y_test, y_pred_list))
    
    average_precision_recall_score = average_precision_score(y_test, y_pred_list)
    
    return average_precision_recall_score

def part_bonus(featurefile, sample_size, hidden_range, nonlinear_fun):
    average_precision_recall_list = []
    for hidden in range(1, hidden_range+1):
        average_precision_recall_score = train_test_model(featurefile, sample_size, hidden, nonlinear_fun)
        average_precision_recall_list.append(average_precision_recall_score)
    plt.plot([hidden for hidden in range(1, hidden_range+1)], average_precision_recall_list)
    plt.xlabel('Hidden Layers')
    plt.ylabel('Average Precision-Recall')
    plt.savefig('avg_precision_recall_hidden_layers.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--sample", type=int, default=1500, help="The sampling size for training data. Test data will be automatically set to sample 1/3 of that number.")
    parser.add_argument("--hidden_range", type=int, default=5, help="The range of the hidden layer size.")
    parser.add_argument("--nonlin", type=str, help="Nonlinear function, either 'relu' or 'tanh'.")
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    
    part_bonus(args.featurefile, args.sample, args.hidden_range, args.nonlin)
