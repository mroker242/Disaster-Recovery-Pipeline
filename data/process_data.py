import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Import data and merge.

    :param messages_filepath: (string) filepath to be imported.
    :param categories_filepath: (string) filepath to be imported.
    :return: dataframe
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge both datasets
    df = pd.merge(messages, categories, on='id')
    
    return df
    

def clean_data(df):
    """
    Perform cleaning techniques on data.

    Create dataframe on the 36 category columns, convert fields to numeric, concatenate categories
    dataframe to main dataframe, drop duplicates.

    :param df: dataframe
    :return: cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[1]

    columns = []
    for var in list(categories.iloc[1]):
        columns.append(var[0:-2])
    # print('Clean data columns: \n')
    # print(columns)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = columns
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str)
        categories[column] = categories[column].str[-1]
    
        # convert column from string to numeric
        #categories[column] = categories[column].astype(int)
        categories[column] = pd.to_numeric(categories[column], errors='coerce')
    
    # drop the original categories column from `df`
    df.drop(columns='categories', inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join='inner')
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    
    return df
    
    


def save_data(df, database_filename):
    """
    Import dataframe and save to specified filename

    :param df: dataframe
    :param database_filename: (string) filename
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)  


def main():
    """
    Retrieve arguments from sys.argv, run load_data to retrieve dataframe, save data.
    """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()