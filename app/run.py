import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Text is transformed to lowercase, stripped, and tokenized.

    :param text: string that contains more than one word
    :return:  array of string text that has been cleaned.
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """ Home page which displays visual of Distribution of Message Genres."""
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    print(df.columns)

    # visualisation showing the top categories
    cat = df.iloc[:, 4:40]
    col_names = []
    # get column names
    for col in cat.columns:
        col_names.append(col)

    # getting totals
    totals = []
    for col in cat.columns:
        totals.append(cat[col].sum())

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

        {

            'data': [
                Bar(
                    x=col_names,
                    y=totals
                )
            ],

            'layout': {
                'title':'Category Totals',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis':{
                    'title': 'Categories',
                    'tickangle': -45
                }
            }

        }


    ]





    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Processes query, runs a prediction on a model, retrieve the classification results and displays them.

    :return: template html page (go.html)
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))



    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    """Run application on host 0.0.0.0 and port 3001"""
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()