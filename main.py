import pandas as pd
from gensim.models import Word2Vec
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics import Rectangle, Color, Line
from kivy.clock import Clock

from CosineSimilarityStrategy import generate_answer
from GRUStrategy import load_GRU_model, generate_gru_answer
from LSTMStrategy import load_lstm_model, generate_lstm_answer
from seq2seqGRUStrategy import clean_text, tokenize, load_model_and_weights, gru_evaluate
from seq2seqLSTMStrategy import load_lstm_seq2seq_model_and_weights, lstm_evaluate

# Prepare data and models
data = pd.read_csv("model_and_weight/chat_health.csv")
data["short_question"] = data["short_question"].apply(clean_text)
data["short_answer"] = data["short_answer"].apply(clean_text)
question = data["short_question"].tolist()
answer = data["short_answer"].tolist()
input_tensor, inp_lang = tokenize(question)
target_tensor, targ_lang = tokenize(answer)
gru_encoder, gru_decoder = load_model_and_weights("model_and_weight/encoder_config.json",
                                          "model_and_weight/decoder_config.json", "model_and_weight/encoder_weights.h5",
                                          "model_and_weight/decoder_weights.h5")
lstm_encoder, lstm_decoder = load_lstm_seq2seq_model_and_weights("model_and_weight/LSTM_encoder_config.json",
                                              "model_and_weight/LSTM_decoder_config.json",
                                              "model_and_weight/LSTM_encoder_weights.h5",
                                              "model_and_weight/LSTM_decoder_weights.h5")
GRU_model = load_GRU_model()
LSTM_model = load_lstm_model()
cosine_similarity_word_model = Word2Vec.load("model_and_weight/word2vec_model_2000_pairs.model")

def ask_cosine_similarity(sentence):
    answer = generate_answer(sentence, cosine_similarity_word_model)
    return answer

def ask_gru(sentence):
    answer = generate_gru_answer(sentence, GRU_model)
    return answer

def ask_lstm(sentence):
    answer = generate_lstm_answer(sentence, LSTM_model)
    return answer

def ask_seq2seq_gru(sentence):
    result, sentence = gru_evaluate(sentence, gru_encoder, gru_decoder, input_tensor, inp_lang, target_tensor, targ_lang)
    return result

def ask_seq2seq_lstm(sentence):
    result, sentence = lstm_evaluate(sentence, lstm_encoder, lstm_decoder, input_tensor, inp_lang, target_tensor, targ_lang)
    return result

class StrategySelectorScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        strategies = {
            'Cosine Similarity': ask_cosine_similarity,
            'Bidirectional GRU': ask_gru,
            'Bidirectional LSTM': ask_lstm,
            'Seq2Seq (GRU)': ask_seq2seq_gru,
            'Seq2Seq (LSTM)': ask_seq2seq_lstm
        }
        for strategy, function in strategies.items():
            btn = Button(text=strategy)
            btn.bind(on_press=lambda instance, func=function, name=strategy: self.select_strategy(func, name))
            layout.add_widget(btn)
        self.add_widget(layout)

    def select_strategy(self, strategy_func, strategy_name, *args):
        chat_interface = self.manager.get_screen('chat_interface')
        chat_interface.selected_strategy = strategy_func
        chat_interface.selected_strategy_name = strategy_name
        self.manager.current = 'chat_interface'

class ChatBotInterface(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        layout = BoxLayout(orientation='vertical')
        self.scroll_view = ScrollView(size_hint=(1, 0.8), do_scroll_x=False)
        self.label = Label(size_hint_y=None, text='Welcome to health Chatbot!', font_name='Roboto', valign='top', halign='left',
                           markup=True, color=(0, 0, 0, 1))
        self.label.bind(width=lambda *x: self.label.setter('text_size')(self.label, (self.label.width, None)))
        self.label.texture_update()
        self.label.height = max(self.label.texture_size[1], 1)
        self.scroll_view.add_widget(self.label)
        with self.scroll_view.canvas.before:
            Color(1, 1, 1, 1)
            self.rect = Rectangle(size=self.scroll_view.size, pos=self.scroll_view.pos)
            Color(0, 0, 0, 1)
            self.line = Line(rectangle=(self.scroll_view.x, self.scroll_view.y,
                                        self.scroll_view.width, self.scroll_view.height), width=1.2)
        self.scroll_view.bind(size=self.update_graphics, pos=self.update_graphics)
        self.input_layout = BoxLayout(size_hint=(1, 0.2))
        self.text_input = TextInput(size_hint=(0.6, 1), font_name='Roboto')
        self.send_button = Button(text='Send', size_hint=(0.2, 1), font_name='Roboto')
        self.return_button = Button(text='Return', size_hint=(0.2, 1), background_color=(1, 0, 0, 1))
        self.send_button.bind(on_press=self.send_message)
        self.return_button.bind(on_press=self.return_to_selector)
        self.input_layout.add_widget(self.text_input)
        self.input_layout.add_widget(self.send_button)
        self.input_layout.add_widget(self.return_button)
        layout.add_widget(self.scroll_view)
        layout.add_widget(self.input_layout)
        self.add_widget(layout)

    def clean_response(self, response):
        # Remove the <eos> token from the response
        return response.replace('<eos>', '')

    def send_message(self, instance):
        user_message = self.text_input.text.strip()
        if user_message:
            response = self.selected_strategy(user_message) if self.selected_strategy else "No strategy selected."
            cleaned_response = self.clean_response(response)  # Clean the response
            strategy_info = f" ({self.selected_strategy_name})" if hasattr(self, 'selected_strategy_name') else ""
            self.label.text += f"\n[color=ff3333]User: {user_message}[/color]"
            self.label.text += f"\n[color=3333ff]Chatbot{strategy_info}: {cleaned_response}[/color]"
            self.text_input.text = ''
            self.label.texture_update()
            self.label.height = self.label.texture_size[1]
            if self.label.height > self.scroll_view.height:
                Clock.schedule_once(self.scroll_to_bottom)

    def update_graphics(self, instance, value):
        self.rect.pos = instance.pos
        self.rect.size = instance.size
        self.line.rectangle = (instance.x + 1, instance.y + 1, instance.width - 2, instance.height - 2)

    def scroll_to_bottom(self, *args):
        self.scroll_view.scroll_y = 0

    def return_to_selector(self, instance):
        self.manager.current = 'strategy_selector'

class ChatBotApp(App):
    def build(self):
        sm = ScreenManager()
        sm.add_widget(StrategySelectorScreen(name='strategy_selector'))
        sm.add_widget(ChatBotInterface(name='chat_interface'))
        return sm

if __name__ == '__main__':
    ChatBotApp().run()
