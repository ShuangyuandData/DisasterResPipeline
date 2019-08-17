import sys

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt','wordnet'])

import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(database_filepath):
    
    enginepath='sqlite:///'+ database_filepath
    
    engine = create_engine(enginepath)
    
    sqlcom='SELECT * FROM '+ database_filepath[-19:-3]
    print(sqlcom)
    
    df = pd.read_sql(sqlcom, engine)
    
    X = df.message
    
    Y = df.iloc[:,4:40]
    
    category_names=list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
        
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)  # Punctuation Removal
    
    tokens=word_tokenize(text)   # tokenization
    
    lemmatizer=WordNetLemmatizer()  
    
    clean_tokens =[]
    
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect',  CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mclf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'mclf__estimator__n_estimators': [10],
        'mclf__estimator__min_samples_split': [2, 3]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    
    output f1 score, precision and recall for the test set
    
    """
    Y_pred=model.predict(X_test)
    
    Y_pred2=pd.DataFrame(Y_pred, index=Y_pred[:,0], columns=category_names)
    
    target_names=['0','1']
    
    print('Evaluation for the model:')
    print(classification_report(Y_test, Y_pred2, target_names=target_names))
        
    return None    


def save_model(model, model_filepath):
    
    modelfile=open(model_filepath,'wb')
    
    pickle.dump(model, modelfile)
    
    modelfile.close()
    
    return None


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        #print('not available')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()