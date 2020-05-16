from flask import Flask, jsonify, request #import objects from the Flask model
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import math
app = Flask(__name__) #define app using Flask

TOKEN_SPAM_PROB_FILE = 'Data/SpamData/03_Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE = 'Data/SpamData/03_Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE = 'Data/SpamData/03_Testing/prob-all-tokens.txt'
VOCAB_SIZE = 2500

def clean_message_no_html(message, stop_words = set(stopwords.words('english')),
                 stemmer = PorterStemmer()):
    filtered_word = []
    
    # clean html text
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()
    
    words = word_tokenize(cleaned_text.lower())
    
    for word in words:
        # remove stopwords and punctuation
        if word not in stop_words and word.isalpha():
            stemmed_word = stemmer.stem(word) 
            filtered_word.append(stemmed_word)
    
    return filtered_word

text_file = open("Data/SpamData/02_Training/word-index.txt", "r")
word_index = text_file.readlines()
text_file.close()
word_index = [x.replace('\n', '') for x in word_index]
# word_index = np.array(map(lambda s: s.strip(), word_index))
type(word_index)

def make_sparse_matrix(df, indexed_words):

    type(indexed_words)
    nr_rows = df.shape[0]
    nr_cols = df.shape[1]
    words_list = set(indexed_words) ## De tim kiem cho nhanh
    # print(type(words_list))
    dict_list = []
    
    for i in range(nr_rows):
        for j in range(nr_cols):
            # print(i, j)
            word = df.iat[i, j]
            if word in words_list:
                doc_id = df.index[i]
                word_id = indexed_words.index(word)
                
                item = ({'WORD_ID': word_id,
                             'OCCURENCE': 1})
                dict_list.append(item)
    
    return pd.DataFrame(dict_list)

def make_full_matrix(sparse_matrix, nr_words, doc_idx=0, word_idx=1, freq_idx=3):

    full_matrix = [0] * 2500
    for i in range(sparse_matrix.shape[0]):
        index = sparse_matrix.WORD_ID.at[i]
        occurent = sparse_matrix.OCCURENCE.at[i]
        full_matrix[index] = occurent

    
    return full_matrix
def spam_or_not(email):
    stemmed_nested_list = clean_message_no_html(email)
    stemmed_nested_list = [stemmed_nested_list]
    word_column_df = pd.DataFrame(stemmed_nested_list)
    sparse_test_df = make_sparse_matrix(word_column_df, word_index)
    test_grouped = sparse_test_df.groupby(['WORD_ID']).sum()
    test_grouped = test_grouped.reset_index()
    full_test_data = make_full_matrix(test_grouped, VOCAB_SIZE)
    prob_token_spam = np.loadtxt(TOKEN_SPAM_PROB_FILE, delimiter=' ')
    prob_token_ham = np.loadtxt(TOKEN_HAM_PROB_FILE, delimiter=' ')
    prob_all_tokens = np.loadtxt(TOKEN_ALL_PROB_FILE, delimiter=' ')
    PROB_SPAM = 0.3116
    joint_log_spam = np.multiply(full_test_data,np.exp((np.log(prob_token_spam) - np.log(prob_all_tokens)) + np.log(PROB_SPAM)))
    joint_log_ham = np.multiply(full_test_data,np.exp((np.log(prob_token_ham) - np.log(prob_all_tokens)) + np.log(1- PROB_SPAM)))

    spam = 1
    ham = 1
    for i in range(2500):
        if joint_log_spam[i] != 0:
            spam = spam * joint_log_spam[i]
        if joint_log_ham[i] != 0:
            ham = ham * joint_log_ham[i]
    print(spam)
    print(ham)
    if spam > ham:
        return "spam"
    else:
        return "ham"


@app.route('/', methods=['GET'])
def test():
	return jsonify({'message' : 'It works!'})
    
@app.route('/', methods=['POST'])
def addOne():
	email = request.json['email']
	return jsonify({'result' : spam_or_not(email)})


if __name__ == '__main__':
	app.run(debug=True, port=8080) #run app on port 8080 in debug mode