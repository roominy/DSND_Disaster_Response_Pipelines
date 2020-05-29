import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import numpy as np

import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import pickle as pkl

def load_data(database_filepath):
    '''
    Load data from database into dataframes of features target
    
    args :
    
        database_filepath (str): filepath of sql database file
    
    return :
    
        X (series): message text data (features)
        y (dataframe): categories (target)
        category_names (index): labels for the categories
        
    '''
    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    table_name = database_filepath.split('/')[-1].split(".")[0] 
    df = pd.read_sql_table(table_name, engine)
    X = df['message']
    y = df.iloc[:,5:]
    category_names = y.columns 
    
    return X, y, category_names
    


def tokenize(text):
    """
    
    Convert messages capitalization case to lower and remove any special characters then lemmatize texts
    
    args:
    
        text (str): the messages text
    
    return:
    
        clean_tokens (list): clean tokenize and lemmatize tokens from messages text 
    
    """ 
    
    # replce urls with a placeholder in message text
    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    found_urls = re.findall(url_pattern,text)
    for url in found_urls:
        text = text.replace(url, "urlplaceholder")
    
    # remove punctuation and tokenize message text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    
    lemmatizer = WordNetLemmatizer()
    
     # lemmatize tokens
    clean_tokens = []
    for token in tokens:    
        clean_token = lemmatizer.lemmatize(token).strip()
        clean_tokens.append(clean_token)
        
    return clean_tokens


def build_model():
    """
    
    Build model with a pipeline with a refined parameters using grid search
    
    return: 
    
    model (pipeline): refind model
    
    """
    
    # build the machine learning pipeline
    pipeline = Pipeline([   ('vect', CountVectorizer(tokenizer=tokenize)),
                            ('tfidf', TfidfTransformer()),
                            ('clf', MultiOutputClassifier(RandomForestClassifier()))
                        ])
    
    # set parameters for grid search  
    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 4]
                 }
    
    # use grid search to find better parameters
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs = -1, verbose= 50)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    
    Disply precision, recall, f1-score that the model scored on the testing set
    
    args: 
    
    model (pipeline): the traind clessifer  
    X_test (dataframe): test set features from message text
    Y_test (dataframe):  test set categories (the target)
    category_names (index): labels for the categories
    
    """
    
    # predict with model
    Y_pred = model.predict(X_test)
    
    # print scores
    print(classification_report(Y_test.values, Y_pred, target_names=category_names))
    


def save_model(model, model_filepath):
    """
    
    Pickle and dump (save) the model to the model_filepath 
    
    args: 
    
    model (pipeline): the traind clessifer 
    model_filepath (str): filepath of wher the model file will be dumped 
    
    """
   
    pkl.dump(model, open(model_filepath, 'wb'))


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