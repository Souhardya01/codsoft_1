import random
import json
import pickle
import numpy as np
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping

lemmatizer = WordNetLemmatizer()

with open('intents.json') as file:
    intents = json.load(file)
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']
for intents in intents['intents']:
    for pattern in intents['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intents['tag']))
        if intents['tag'] not in classes:
            classes.append(intents['tag'])
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([i[0] for i in training])
train_y = np.array([i[1] for i in training])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(train_x, train_y, epochs=200, batch_size=8, callbacks=[early_stop])
model.save('chatbot_model.h5')
print("Model training complete and saved.")

def clean_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
def bag_of_words(sentence, words):
    sentence_words = clean_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)
def chat():
    print("Start talking with the bot! (type 'quit' to stop)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        bow = bag_of_words(user_input, words)
        result = model.predict(np.array([bow]))[0]
        tag = classes[np.argmax(result)]

        for intents in intents['intents']:
            if intents['tag'] == tag:
                responses = intents['responses']
                print(f"Bot: {random.choice(responses)}")
