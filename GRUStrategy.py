
from keras import Sequential
from keras.src.layers import Embedding, GRU, Dense, Dropout, Bidirectional, BatchNormalization
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import string
import unicodedata



# Define function to convert unicode to ASCII
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

# Define text cleaning function
def clean_text(text):
    text = unicode_to_ascii(text.lower().strip())
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub("(\\W)"," ",text)

    return text



def generate_gru_answer(question,model):
    with open('model_and_weight/tokenizer_GRU_2000.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    question_seq = tokenizer.texts_to_sequences([question])
    question_padded = pad_sequences(question_seq, maxlen=57, padding='post')
    prediction = model.predict(question_padded)
    predicted_indices = np.argmax(prediction, axis=-1)[0]
    predicted_words = ' '.join([tokenizer.index_word[i] for i in predicted_indices if i != 0])
    return predicted_words



def load_GRU_model():
    GRU_model = Sequential([
        Embedding(6182, 128),
        Bidirectional(GRU(256, return_sequences=True)),
        Dropout(0.5),
        BatchNormalization(),
        Bidirectional(GRU(256, return_sequences=True)),
        Dropout(0.5),
        Dense(6182, activation='softmax')
    ])
    GRU_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    GRU_model.build(input_shape=(None, 62))
    GRU_model.load_weights('model_and_weight/GRU_2000.weights.h5')
    return GRU_model


