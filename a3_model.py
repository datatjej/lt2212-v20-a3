import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
# Whatever other imports you need
import seaborn as sns

# You can implement classes and helper functions here too.

def split_df(df):
    #divide the input df into a test df and a train df by splitting the df where the 'test' starts:
    #test_row_starting_index = df.loc[df['Type'] == 'test']
    train_df = df.iloc[:2343]
    test_df = df.iloc[2343:]
    #return train_df.to_csv("train_df.csv", index=False), test_df.to_csv("test_df.csv", index=False) 
    return train_df, test_df
    
    df1 = datasX.iloc[:, :72]
df2 = datasX.iloc[:, 72:]
    d = dt.sample(n=2, random_state=rd)  #sample() returns  a random sample of items from an axis of object
                                        #n=number of items from axis to return
    doc1 = d.iloc[0].values.tolist()
    doc2 = d.iloc[1].values.tolist()
    
    if doc1[-1] == doc2[-1]: 
        return (doc1[:-1], doc2[:-1], 1)
    else:
        return (doc1[:-1], doc2[:-1], 0)
        
def build_samples(dt, n= 1000):
    sample_0 = []
    sample_1 = []

    while len(sample_1) < n or len(sample_0) <n:
        s = sample(dt)
        if s[2] == 1 and s not in sample_1 and len(sample_1) < n:
            sample_1.append(s)
        elif s[2] == 0 and s not in sample_0 and len(sample_0) < n:
            sample_0.append(s)
    return sample_0 + sample_1



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    # Add options here for part 3 -- hidden layer and nonlinearity,
    # and any other options you may think you want/need.  Document
    # everything.
    
    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    df = pd.read_csv("df.csv")
    #print(df.head()) #the head() method is used to return top n (5 by default) rows of a data frame or series.
    #sns.countplot(x = 'Class_att', data=df)
    
    

    # implement everything you need here
    
