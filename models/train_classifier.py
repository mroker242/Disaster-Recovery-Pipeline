import sys
# import libraries
import pandas as pd
import numpy as np
import re
import nltk
nltk.download(['punkt', 'wordnet'])
from sklearn.metrics import confusion_matrix
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import pickle
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='messages', con=engine)
    df.related = df.related.replace(2,1)
    # trying this
    df.drop(columns='related', axis=1, inplace=True)
    df.message = df.message.astype(str)
    for i in df.iloc[:,4:40]:
        df[i] = df[i].astype(int)
        #print(df[i].value_counts())
    X = df.message.values
    y = df.iloc[:,4:40].values
    
 
    cat_names = []
    for col in df.iloc[:,4:].columns:
        cat_names.append(col)
        
    return X, y, cat_names


def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('mltoclf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
   
        #'mltoclf__estimator__min_samples_split': [2, 3, 4],
        'mltoclf__estimator__criterion': ['entropy'] #['entropy', 'gini']
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    predicted = model.predict(X_test)
    accuracy = (predicted == Y_test).mean()
    
    print('accuracy: ', accuracy)
    print(classification_report(Y_test, predicted, target_names=category_names))
          
          
          
def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))
          
          
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print('X_train:', X_train.shape)
        print('Y_train:', Y_train.shape)
        print('X_test:', X_test.shape)
        print('Y_test:', Y_test.shape)
 
        
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