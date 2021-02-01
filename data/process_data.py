import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Function to load the two datasets for messages and categories, merge them add the categories separately.
   
        Parameters:
            messages_filepath (str): file path to messages csv file
            categories_filepath (str): file path to categories csv file
        
        Returns:
            df (DataFrame): Dataframe with the two input files merged
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, how='outer', on=['id'])
    categories = categories['categories'].str.split(';', expand=True)

    category_colnames = list(categories.iloc[0].apply(lambda x: x[:-2]))
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1:])
        categories[column] = pd.to_numeric(categories[column])

    df.drop(columns=['categories'], inplace=True)
    df = pd.concat([df, categories], axis=1)

    return df


def clean_data(df):
    '''
    Function to clean DataFrame (df) by removing duplicates.
    '''
    df = df[df['related'].notnull()]
    df['related'] = np.where(df['related'] == 2, 1, df['related'])
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    '''
    Function to save the DataFrame to a SQL database.
    '''  
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages_and_Categories', engine, index=False, if_exists='replace')


def main():
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

