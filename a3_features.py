import os
import sys
import argparse
import numpy as np
import pandas as pd
# Whatever other imports you need
import csv
from glob import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

def data_load(folders):
    corpus = []
    authors= []
    for author_folder in folders:
        author_specific_emails = glob("{}/*".format(author_folder))
        author = author_folder[13:]
        for email_path in author_specific_emails:
            authors.append(author)
            email_content = ""
            with open(email_path, "r") as email:
                for line in email:
                    email_content += line
            corpus += [email_content.lower()]
    return corpus, authors

def vectorize(corpus):
    stop = set(stopwords.words('english'))
    vectorizer = TfidfVectorizer(stop_words=stop, token_pattern=r'(?u)\b[A-Za-z][A-Za-z]+\b')
    vectorized_corpus = vectorizer.fit_transform(corpus)
    return vectorized_corpus #returns 

def reduce_dims(vectorized_corpus, dims):
    svd = TruncatedSVD(n_components=dims)
    reduced_vectorized_corpus = svd.fit_transform(vectorized_corpus)
    return reduced_vectorized_corpus

def spit_in_test_train(X, y, testsize):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize, random_state=42)
    return X_train, X_test, y_train, y_test

def create_df(X_train, X_test, y_train, y_test):
    train_df = pd.DataFrame(X_train)
    train_df.insert(0, "Type", "train", True)
    train_df.insert(0, "Author", y_train, True)
    
    test_df = pd.DataFrame(X_test)
    test_df.insert(0, "Type", "test", True)
    test_df.insert(0, "Author", y_test, True)

    df = pd.concat([train_df, test_df])
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

    folders = glob("{}/*".format(args.inputdir))
    corpus, authors = data_load(folders)
    vectorized_corpus = vectorize(corpus)
    reduced_vectorized_corpus = reduce_dims(vectorized_corpus, args.dims)
    
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    X_train, X_test, y_train, y_test = spit_in_test_train(reduced_vectorized_corpus, authors, args.testsize)
    df = create_df(X_train, X_test, y_train, y_test)
    
    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    output_file = df.to_csv(args.outputfile, index=False)

    print("Done!")
