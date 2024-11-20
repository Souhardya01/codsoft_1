import numpy as np
import random
import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatBot_Model.h5')

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
def words_bag(sentence):
    sentence_words = clean_sentence(sentence)
    bag = [1 if word in sentence_words else 0 for word in words]
    return np.array(bag)
def predict_class(sentence):
    bag = words_bag(sentence)
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [{'intents': classes[r[0]], 'probability': str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    if intents_list:  # To ensure there's a predicted intents
        tag = intents_list[0]['intents']
        for intents in intents_json['intents']:
            if intents['tag'] == tag:
                return random.choice(intents['responses']) # Fallback response
    return "I'm sorry, I didn't understand that. Can you rephrase?"

print("Go! ChatBot is activated. Type 'quit' to exit.")
while True:
    message = input("You: ")
    if message.lower() == 'quit':
        print("ChatBot: Goodbye! Have a great day!")
        break
    predicted_intents = predict_class(message)
    response = get_response(predicted_intents, intents)
    print("ChatBot:", response)
