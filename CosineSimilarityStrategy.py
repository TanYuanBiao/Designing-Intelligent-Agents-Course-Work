import pandas as pd
import numpy as np
import h5py
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity


import time
# Pre-compiled regular expressions
html_tags_regex = re.compile(r'<.*?>')
non_alpha_numeric_regex = re.compile(r'[^a-zA-Z\s]')
stop_words = set(stopwords.words('english'))
#Text Preprocessing
def preprocess_text(text):
    text = html_tags_regex.sub('', text)
    text = non_alpha_numeric_regex.sub('', text)
    text = text.lower()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

def get_question_vector(words, model):
    return np.mean([model.wv[word] for word in words if word in model.wv], axis=0)

def generate_answer(question, model):
    question_words = preprocess_text(question)
    question_vec = get_question_vector(question_words, model)
    with h5py.File('model_and_weight/question_vectors_2000_pairs.h5', 'r') as f:
        vectors = f['vectors'][:]
        cosine_similarities = cosine_similarity([question_vec], vectors)
        best_index = np.argmax(cosine_similarities)
        best_answer = f['answers'][best_index].decode('utf-8')
    return best_answer


# if __name__ == '__main__':
#     cosine_similarity_word_model = Word2Vec.load("model_and_weight/word2vec_model_2000_pairs.model")
#
#     answer = generate_answer("is all vitamin d the same", cosine_similarity_word_model)
#
#     print(answer)