from flask import Flask, request, render_template
import nltk
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
stemmer = LancasterStemmer()
import numpy as np
import random
import json

app = Flask(__name__)
@app.route("/", methods = ["GET", "POST"])
def first():
    chat = ""
    rep = ""
    global data, words, label, patterns_x, tags_y, training
    
    with open('src/shabir.json') as file:
        data = json.load(file)

        words  = []
        labels = []
        patterns_x = []
        tags_y = []
        training = []
        output = []
     
        stop_words = set(stopwords.words('english'))
        
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                
                filtered_wrds = [w for w in wrds if not w.lower() in stop_words]
                filtered_wrds = [stemmer.stem(word.lower()) for word in filtered_wrds]
                
                words.extend(filtered_wrds)
                patterns_x.append(filtered_wrds)
                tags_y.append(intent["tag"])
            
            if intent["tag"] not in labels:
                labels.append(intent["tag"])
                
        words = sorted(list(set(words)))
        labels = sorted(labels)
        
        out_empty = [0 for _ in range(len(labels))]

        for x, p in enumerate(patterns_x):
            bag = []
            for w in words:
                if w in p:
                    bag.append(1)
                else:
                    bag.append(0)            
            output_row = out_empty[:]
            output_row[labels.index(tags_y[x])] = 1
            
            training.append(bag)
            output.append(output_row)
            
        training = np.array(training)
        output = np.array(output)
        
    return render_template('chat.html',ch=chat,rep=rep)

@app.route('/chat', methods = ['GET', 'POST'])
def hello():
    a_chat = request.form.get('a_chat')
    rep = do_chat(a_chat) 
    return render_template('chat.html', chs = a_chat, rep=rep)

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in s_words if not w.lower() in stop_words]
    filtered_sentence = []
    for w in s_words:
        if w not in stop_words:
            filtered_sentence.append(w)
            
    for se in filtered_sentence:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return bag

def do_chat(inp):
    results = ([bag_of_words(inp, words)])
    # print('results',results)
    i=0
    bg = [0 for _ in range(len(words))]
        
    for myList in [results]:
        for item in myList:
            for itt in item:
                bg[i] = itt
                i += 1
    
    no_patterns = 0
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            no_patterns += 1
            
    score = [0 for _ in range(no_patterns)]
        
    k=0
    for t in training:
        j=0
        for i in t:
            if bg[j]==i  and i==1:
                score[k] += 1
            j += 1
        k += 1
      
    results_index = np.argmax(score)      
    
    tag = tags_y[results_index]
    tag2= " ".join(patterns_x[results_index])
    sc = score[results_index]
    
    i=0
    for s in score:
        print('\033[1m',tags_y[i],'\033[0m'," ".join(patterns_x[i]),' score',s)
        i += 1
    
    print('\033[1m',tag, '\033[0m[',tag2,'] score: ',sc)
    
    if sc > 0 :
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses'] 
        rep = random.choice(responses) 
    else :
        rep = 'Sorry, my hearing is bad hehe'
    return rep

if __name__ == '__main__':
    app.run(debug = False)