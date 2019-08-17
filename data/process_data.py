import sys
from sqlalchemy import create_engine
import pandas as pd

def load_data(messages_filepath, categories_filepath):
    
    """
    The function loads the data and performs the first preprocess e.g. merge
    """
    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    
    df=messages.merge(categories, how='outer', on='id')
    
    categories = df.categories.str.split(";", expand=True)
    
    category_colnames = categories.iloc[0].apply(lambda x : x[:-2])
    
    categories.columns = category_colnames
    
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1:])
    
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    # convert 2 in related to 1
    
    categories.replace(2,1,inplace =True)
        
    df_drop=df.drop('categories',axis=1)
    
    df_con = pd.concat([df_drop, categories], axis=1)
    
    return df_con


def clean_data(df):
    
    """
    The function removes duplicates in the data
    """

    print(df.info, '1')
    
    df=df.drop_duplicates(['id'])
    
    print(df.info, '2')
    
    return df


def save_data(df, database_filename):
    
    """
    The function saves the clean data into an sqlite database
    """
    
    database_filename2=database_filename[-19:-3]
    
    print(database_filename2)
    
    enginepath='sqlite:///'+database_filename
    
    engine = create_engine(enginepath)
    
    df.to_sql(database_filename2, engine, index=False)  ### only filename, no path
    
    return None
    


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