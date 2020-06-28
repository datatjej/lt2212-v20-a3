import os
import sys
import argparse
import numpy as np
import pandas as pd
# Whatever other imports you need
import csv
from glob import glob
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

def data_load(inputdir):
    #get subdirectories of inputdir with help of the glob module:
    folders = glob("{}/*".format(inputdir))
    dict_emailindex_word = {}
    stop_words = set(stopwords.words('english'))
    for author_folder in folders:
        with open
        emails = []
        # all words in author_folder tokenized and saved in list:
        words += [word.lower() for word in sample.split() if (word.isalpha() and word not in stop_words)]
        ## filter out unique words and their frequency in the sample:
        uniqueWords, wordCount=get_unique(words)
        
        sample_index = "Doc_" + str(samples.index(sample))
        for index, count in enumerate(wordCount):
            if sample_index in dict_postindex_word:
                dict_postindex_word[sample_index][uniqueWords[index]]=count
            else: 
                dict_postindex_word[sample_index]={}
                dict_postindex_word[sample_index][uniqueWords[index]]=count
    
    #fill out NaN cells with 0's:
    df = pd.DataFrame(dict_postindex_word).fillna(0) 
    #transpose the dateframe so that x-axis becomes y-axis and vice versa: 
    df_transposed = df.T
    #turn df into numpy array:
    df_as_nparray = df_transposed.to_numpy()
    
    freq_sums_of_nparray = np.sum(df_as_nparray, axis =0)
    filtered_nparray = freq_sums_of_nparray > 10
    features = df_as_nparray[:, filtered_nparray]
    
    return features
    

def get_unique(x):
    y, f = np.unique(x, return_counts=True)
    return y, f

    
	
    with open(path_to_dataset, "r") as thefile:
        for line in thefile:
            line_lists.append(line)
    
    random.shuffle(line_lists)
    
    train_dataset = line_lists[0:round(len(line_lists)*0.8)]
    test_dataset = line_lists[round(len(line_lists)*0.8):]
    
    train_df = create_csv(train_dataset,'train.csv')
    test_df = create_csv(test_dataset,'test.csv')
    
    return train_df,test_df
    
def split_example(example):
    line_split = example.split()
    
    word_sense = line_split[0]
    word_form = line_split[1]
    word_index = line_split[2]
    
    return word_sense,word_form,word_index,line_split[3:]
    
    
def create_csv(dataset,csv_name):
    word_sense_list = []
    word_form_list = []
    word_index_list = []
    tokenized_context_list = []
    
    for example in dataset:
        word_sense,word_form,word_index,tokenized_context = split_example(example)
        
        word_sense_list.append(word_sense)
        word_form_list.append(word_form)
        word_index_list.append(word_index)
        tokenized_context_list.append(tokenized_context)
        
    d = {'word_sense': word_sense_list, 'word_form': word_form_list, 'word_index': word_index_list, 'tokenized_context': tokenized_context_list}
    df = pd.DataFrame(data=d)
#     df.to_csv(csv_name)
    
    return df
	

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.

    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    
    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.

    print("Done!")
    
