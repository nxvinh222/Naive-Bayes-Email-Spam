from flask import Flask, jsonify, request #import objects from the Flask model
import numpy as np
import pandas as pd
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
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
    """
    Return Sparse matrix as dataframe
    
    df: Dataframe with words in columns with document id as index (X_train or X_test)
    indexed_words: index of words ordered by word id
    labels: category as a series (y_train or y_test)
    """
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
    """
    From a full matrix from a sparse matrix.
    Return pandas DataFrame
    Keyword arguments:
    sparse_matrix -- numpy array
    nr_words -- size of vocabulary, total number of tokens
    doc_idx -- position of the document id in the sparse matrix
    word_idx -- position of the word id in the sparse matrix
    cat_idx -- position of the label (spam 0, nonspam 1) in the sparse matrix
    freq_idx -- position of the occurence in the sparse matrix
    
    """
    # doc_id_names = np.unique(sparse_matrix[:, 0])
    # column_names = list(range(0, VOCAB_SIZE))
    # full_matrix = pd.DataFrame(columns=column_names)
    # full_matrix.fillna(value=0, inplace=True)
    
    # doc_id = sparse_matrix[doc_idx]
    # word_id = sparse_matrix[word_idx]
    # occurence = sparse_matrix[freq_idx]
        
    # full_matrix.at[doc_id, 'DOC_ID'] = doc_id
    # full_matrix.at[doc_id, word_id] = occurence
        
    # full_matrix.set_index('DOC_ID', inplace=True)
    full_matrix = [0] * 2500
    for i in range(sparse_matrix.shape[0]):
        index = sparse_matrix.WORD_ID.at[i]
        occurent = sparse_matrix.OCCURENCE.at[i]
        full_matrix[index] = occurent

    
    return full_matrix
def spam_or_not(email):
    stemmed_nested_list = clean_message_no_html(email)
    # print(stemmed_nested_list)
    stemmed_nested_list = [stemmed_nested_list]
    word_column_df = pd.DataFrame(stemmed_nested_list)
    # print(word_column_df)
    sparse_test_df = make_sparse_matrix(word_column_df, word_index)
    # print(sparse_train_df)
    # index = np.loadtxt('Data/SpamData/02_Training/word-index.txt', delimiter=',')
    test_grouped = sparse_test_df.groupby(['WORD_ID']).sum()
    test_grouped = test_grouped.reset_index()
    # print(test_grouped.WORD_ID.shape)
    full_test_data = make_full_matrix(test_grouped, VOCAB_SIZE)
    # print(test_grouped)
    prob_token_spam = np.loadtxt(TOKEN_SPAM_PROB_FILE, delimiter=' ')
    prob_token_ham = np.loadtxt(TOKEN_HAM_PROB_FILE, delimiter=' ')
    prob_all_tokens = np.loadtxt(TOKEN_ALL_PROB_FILE, delimiter=' ')
    # test_data = pd.DataFrame(full_test_data).reshape(2500,)
    # print(pd.DataFrame(full_test_data).sum())
    # print(np.multiply(full_test_data, prob_token_spam))
    PROB_SPAM = 0.3116
    joint_log_spam = np.multiply(full_test_data,(np.log(prob_token_spam) - np.log(prob_all_tokens)) + np.log(PROB_SPAM))
    joint_log_ham = np.multiply(full_test_data,(np.log(prob_token_ham) - np.log(prob_all_tokens)) + np.log(1- PROB_SPAM))
    
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

@app.route('/lang', methods=['GET'])
def returnAll():
	return jsonify({'languages' : languages})

@app.route('/lang/<string:name>', methods=['GET'])
def returnOne(name):
	langs = [language for language in languages if language['name'] == name]
	return jsonify({'language' : langs[0]})

@app.route('/', methods=['POST'])
def addOne():
	email = request.json['email']
	return jsonify({'result' : spam_or_not(email)})


if __name__ == '__main__':
	app.run(debug=True, port=8080) #run app on port 8080 in debug mode