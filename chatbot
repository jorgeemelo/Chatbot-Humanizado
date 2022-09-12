import json
import numpy as np
import pickle
import random
from colorama import Fore, Style, Back
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import colorama
from keras.utils.data_utils import pad_sequences

colorama.init()

with open("./intents_data/intents.json") as file:
    data = json.load(file)

def chat():
    # load trained model
    model = keras.models.load_model('chat_model')

    # load tokenizer object
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "Usuario: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "Bot:" + Style.RESET_ALL, np.random.choice(i['responses']))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

print(Fore.YELLOW + "Começe a interagir com o robô! (para encerrar a conversa, escreva *SAIR*" + Style.RESET_ALL)
chat()
