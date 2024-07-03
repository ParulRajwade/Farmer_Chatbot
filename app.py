import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from translate import Translator

from keras.models import load_model
model = load_model('A:/Work/1. MapMyCrop Work/AgroBot/demo_app/chatbot_model.h5')

import json
import random
intents = json.loads(open('A:/Work/1. MapMyCrop Work/AgroBot/demo_app/intents.json',encoding = "utf-8").read())
words = pickle.load(open('A:/Work/1. MapMyCrop Work/AgroBot/demo_app/words.pkl','rb'))
classes = pickle.load(open('A:/Work/1. MapMyCrop Work/AgroBot/demo_app/classes.pkl','rb'))

############ Some functions ############

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


############ Some more functions ############

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


############ Linguistics ############

class difflanguage:
    def lang(self, Language):

        default = "English"

        return getattr(self, 'case_' + str(Language), lambda: default)()

    def case_1(self):
        translator= Translator(from_lang="english",to_lang=("gujarati"))
        translation = translator.translate("Good Morning")
        return translation

 

    def case_2(self):
        translator= Translator(from_lang="english",to_lang=("marathi"))
        translation = translator.translate("Good Morning")
        return translation



    def case_3(self):
        translator= Translator(from_lang="english",to_lang=("german"))
        translation = translator.translate("Good Morning")
        return translation

 

    def case_4(self):
        translator= Translator(from_lang="english",to_lang=("french"))
        translation = translator.translate("Good Morning")
        return translation



    def case_5(self):
        translator= Translator(from_lang="english",to_lang=("tamil"))
        translation = translator.translate("Good Morning")
        return translation
    
    
   
my_switch = difflanguage()
my_switch.lang(1)


############ Some more functions ############

import tkinter as tk
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)
    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="Black", font=("Verdana", 14))
        res = chatbot_response(msg)
        my_switch = difflanguage()
        ChatLog.insert(END, "AgBot: " + res + '\n\n')
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)

base = tk.Tk()
base.title("Multilingual Chatbot")
base.geometry("700x800")
base.resizable(width=FALSE, height=FALSE)
ChatLog = Text(base, bd=0, bg="#8E8D8D", height="10", width="60", font="Arial")
ChatLog.config(state=DISABLED)
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set
SendButton = Button(base, font=("Verdana",18,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#5CA1CB", activebackground="#3c9d9b",fg='#ffffff',
                    command= send )
EntryBox = Text(base, bd=0, bg="WHITE",width="29", height="5", font=("Verdana",18))
scrollbar.place(x=682,y=6, height=800)
ChatLog.place(x=6,y=6, height=673, width=800)
EntryBox.place(x=301, y=679, height=120, width=380)
SendButton.place(x=6, y=679, height=120,width=300)
base.mainloop()