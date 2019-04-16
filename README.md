# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
# Disaster-Recovery-Pipeline

In the event of a disaster, and in an effort to get help, persons send messages in order to get emergency
help. This application uses an ETL model, a machine learning model to import messages, use NLP 
machine learning algorithms to classify the messages so that the relevant departments or organizations
can assist when needed. Please follow below for installation instructions and dependencies. 



##Installation

- To run the application, navigate to the `app` directory and execute `python run.py`. This will start 
the application.
- To open the web app on the browser, go to **http//0.0.0.0:3001/**.
- For usage, type in text as required.


##Dependencies

- json
- plotly
- pandas
- word_tokenize
- WordNetLemmatizer
- Flask (render_template, jsonify, request)
- sqlalchemy
- Python 3.6


##File Descriptions

- **run.py**: script to execute application.
- **train_classifier.py**: script to train classifier using MultiOutputClassifier.
- **process_data.py**: ETL (Extract Transform Load) script. Imports both `disaster_categories.csv` and
`disaster_messages.csv`, merges and cleans datasets and extracts to `DisasterResponse.db`.


