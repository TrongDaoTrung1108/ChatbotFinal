import json
import pickle
import nltk
import tflearn
import tensorflow
import  random
import numpy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

'''model.load("model.tflearn")'''
model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
model.save("model.tflearn")
with open("intents.json", encoding="utf-8") as file:
    data = json.load(file)
patterns = [i['patterns'] for i in data['intents']]
updated_patterns = []
for pattern in data['intents']:
    updated_patterns += pattern['patterns']
# for pattern in patterns:
#     updated_patterns += pattern
print(updated_patterns)
# print(patterns)
def get_intents_dictionary(intents_data):
    intents_dic = {}
    responses_dic = {}
    for data in intents_data['intents']:
        for pattern in data['patterns']:
            intents_dic[pattern] = data['tag']
        responses_dic[data['tag']] = data['responses']

    return intents_dic, responses_dic


intents_dic, responses_dic = get_intents_dictionary(data)
print(responses_dic)

def get_response_by_question(question, intents_dic, responses_dic):
    tag = intents_dic[question]
    responses = responses_dic[tag]
    num_rand = len(responses)

    return responses[random.randint(0,num_rand-1)]
print(get_response_by_question("Xin chào",intents_dic,responses_dic))
# while True:
#     question = input("Nhap cau hoi: ")
#     print(".......")
#     print(get_response_by_question(question, intents_dic, responses_dic))

    # sentences = ["Machine learning is great", "Natural Language Processing is a complex field",
    #              "Natural Language Processing is used in machine learning"]
    # vocabulary = tokenize_sentences(sentences)
def tokenize_sentences(sentences):
    words = []
    for sentence in sentences:
        w = remove_stopword(sentence)
        words.extend(w)
    words = sorted(list(set(words)))
    return words

def remove_stopword(output):
    stopword = set(stopwords.words('stopword'))
    word_tokens = word_tokenize(output)
    list_stop_word_stop = [w for w in output if w not in stopword]

    list_stop_word_stop = []

    for w in word_tokens:
        if w not in stopword:
            list_stop_word_stop.append(w)
    return list_stop_word_stop

###chuẩn bị data cho model, chuyển data về dạng Bag of Words,TF
import numpy as np
import re

#bag_of_word :
def bagofwords(sentence, words):
    sentence_words = remove_stopword(sentence)
    # frequency word count
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] += 1

    return np.array(bag)

vocabulary = tokenize_sentences(updated_patterns)
def respone_question(question):
    remove_stopword(question)

    results = model.predict([ bagofwords(question,remove_stopword(question))])
    results_index = numpy.argmax(results)
    answer =  get_response_by_question(results_index, intents_dic, responses_dic)
    return answer

print('Chatbot Lily')
while True:
    question = input("Nhập câu hỏi của bạn. Nếu muốn dừng bạn hãy nói 'bye' ")
    if question.strip()!= 'bye':
        # perdiction=PredictQuestion(question)
        # answer = answer_chatbot(perdiction)
        print('Chatbot Answer:',respone_question(question))

    if question.strip()== 'bye':
        print('Chatbot Answer: Bye bye')
        break
