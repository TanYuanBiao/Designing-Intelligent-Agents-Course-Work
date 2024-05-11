
import tensorflow as tf
import re
import string
import unicodedata
from tensorflow.python.keras.models import model_from_json


@tf.keras.utils.register_keras_serializable()
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x,hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))



@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # query_with_time_axis shape == (batch_size, 1, hidden size)
        # values shape == (batch_size, max_len, hidden size)
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights

def load_model_and_weights(encoder_config_path, decoder_config_path, encoder_weights_path, decoder_weights_path):
    # 检查 Encoder 的 JSON 配置
    with open(encoder_config_path, 'r') as file:
        encoder_config = file.read()

    # 检查 Decoder 的 JSON 配置
    with open(decoder_config_path, 'r') as file:
        decoder_config = file.read()

    # 加载模型结构
    encoder = model_from_json(encoder_config, custom_objects={
        'Encoder': Encoder,
        'GRU': tf.keras.layers.GRU
    })

    decoder = model_from_json(decoder_config, custom_objects={
        'Decoder': Decoder,
        'GRU': tf.keras.layers.GRU,
        'BahdanauAttention': BahdanauAttention
    })

    # 输出模型类型
    print("Type of encoder:", type(encoder))
    print("Type of decoder:", type(decoder))
    # 创建假输入数据来初始化模型
    encoder_dummy_input = tf.random.uniform((64, 13), dtype=tf.float32)  # 输入长度为13，批量大小为64
    encoder_hidden_initial = encoder.initialize_hidden_state()
    encoder_output, encoder_hidden = encoder(encoder_dummy_input, encoder_hidden_initial)  # 初始化并构建模型

    # 为decoder创建正确的假输入
    decoder_dummy_input = tf.random.uniform((64, 1), dtype=tf.float32)  # 每次解码只处理一个时间步长的输入
    decoder_output, decoder_hidden, _ = decoder(decoder_dummy_input, encoder_hidden, encoder_output)  # 构建模型

    # 加载权重
    encoder.load_weights(encoder_weights_path)
    decoder.load_weights(decoder_weights_path)

    return encoder, decoder
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
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub("(\\W)"," ",text)
    text = re.sub('\S*\d\S*\s*','', text)
    text =  "<sos> " +  text + " <eos>"
    return text
def remove_tags(sentence):
    return sentence.split("<start>")[-1].split("<end>")[0]

def tokenize(lang):
  # Create a tokenizer object, specifying an empty string for filters to keep all characters
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    # Fit the tokenizer on the provided text data
    lang_tokenizer.fit_on_texts(lang)
    # Convert the text data into sequences of integers
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # Pad the sequences so they all have the same length for model training
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')

    return tensor, lang_tokenizer

def tokenize(lang):
  # Create a tokenizer object, specifying an empty string for filters to keep all characters
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
    # Fit the tokenizer on the provided text data
    lang_tokenizer.fit_on_texts(lang)
    # Convert the text data into sequences of integers
    tensor = lang_tokenizer.texts_to_sequences(lang)
    # Pad the sequences so they all have the same length for model training
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,padding='post')

    return tensor, lang_tokenizer
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')
def gru_evaluate(sentence,encoder,decoder,input_tensor,inp_lang,target_tensor,targ_lang):

    max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
    sentence = clean_text(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],maxlen=max_length_inp,padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, 1024))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<sos>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,dec_hidden,enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<eos>':
            return remove_tags(result), remove_tags(sentence)

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return remove_tags(result), remove_tags(sentence)
