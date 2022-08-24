#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, request, redirect, url_for
import pickle
import joblib

from sklearn.feature_extraction.text import CountVectorizer


def split_into_words(i):
    return (i.split(" "))

#filename='nlp_model.pkl'
classifier_mb= joblib.load(open('nlp_model1.pkl','rb'))
f = open('mail_vector.pkl', 'rb')
mail_vector = pickle.load(f)
f.close()

app=Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    result_data = "none"
    text_data = "none"
    if 'result_data' in request.args:
        result_data = request.args['result_data']
    if 'text_data' in request.args:
        text_data = request.args['text_data']
    return render_template('index.html', result_data=result_data, text_data=text_data)

@app.route('/process', methods=['POST'])
def process():
    message= request.form['text_data']
    data= [message]
    vect= mail_vector.transform(data).toarray()
    my_prediction= classifier_mb.predict(vect)
    return redirect(url_for('index', result_data=my_prediction[0], text_data=message))
    
if __name__=='__main__':
    app.run(debug=False)


# In[ ]:


get_ipython().run_line_magic('tb', '')


# In[ ]:


import sys
print(sys.executable)


# In[ ]:




