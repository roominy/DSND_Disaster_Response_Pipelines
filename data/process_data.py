import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    
    Loads  messages and categories then merges then to create dataframe
    
    args :
    
    messages_filepath (str) : filepath for messages csv file
    categories_filepath (str): filepath for categories csv file
    
    return :
    
    df (dataframe): merged messages and categories datafreame
    
    
    """
    
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on='id')
    return df
    


def clean_data(df):
    """
    
    Performs the following: 
    - Split categories into separate category columns to create df with new category columns
    - Convert category values to just numbers 0 or 1
    - Replace categories column in df with new category columns
    - Remove duplicates
    
    args :
    
    df (dataframe): merged messages and categories datafreame
    
    
    return :
    
    df (dataframe): cleaned dataframe 
    
    
    """
    
    
    ##Split categories into separate category columns
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(pat=';',expand=True)
    
    # categories into separate category columns
    row = categories.loc[0]
    
    # extract a list of new column names for categories.
    category_colnames = row.apply(lambda x:x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames 
    
    
    ## Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    ## Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace = True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    ## Remove duplicates
    # drop duplicates
    df.drop_duplicates(inplace = True)
    
    return df
    
    


def save_data(df, database_filename):
    """
    
    Loads dataframe and save it in SQLite database
    
    args :
    
    df (dataframe): cleaned dataframe 
    
    """
   
    engine = create_engine('sqlite:///{}'.format(database_filename))
    table_name = database_filename.split('/')[-1].split(".")[0]
    df.to_sql(table_name, engine,if_exists='replace', index=False) 


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