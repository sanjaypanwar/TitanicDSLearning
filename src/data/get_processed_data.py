import numpy as np
import pandas as pd
import os

def read_data():
    #set the path of the raw data
    raw_data_patch = os.path.join(os.path.pardir,'data','raw')
    train_file_data = os.path.join(raw_data_patch, 'train.csv')
    test_file_path = os.path.join(raw_data_patch, 'test.csv')
    # read the data with all default parameters
    train_df = pd.read_csv(train_file_data, index_col='PassengerId')
    test_df = pd.read_csv(test_file_path, index_col= "PassengerId")
    test_df['Survived']=-888
    df=pd.concat((train_df,test_df),axis=0)
    return df

def process_data(df):
    # here we could have all the script in this notebook previously 
    # but rather then copying hundreds of lines above, we can use one of the very useful technique 'method chaining'
    # 'mehtod chaing' -- concept where we can use one method and pass the result to next method in chain
    
    return(df
           # create title attribute - then add this
           # below x refers to the data frame, unlike above
           .assign(Title= lambda x : x.Name.map(get_title))
           # working missing values - start with this
           # 'pipe' function we use to apply any function to dataframe
           .pipe(fill_missing_values)
           #create fare bin feature
           .assign(Fare_bin = lambda x : pd.qcut(x.Fare,4,labels=['very_low','low','high','very_high']))
           #create age state
           .assign(AgeState = lambda x : np.where(x.Age>=18,'Adult','Child'))
           .assign(FamilySize = lambda x : x.Parch +x.SibSp +1)
           .assign(IsMother = lambda x : np.where(((x.Sex=='female') & (x.Age>18) & (x.Title!='Miss') & (x.Parch>0)), 1, 0))
           # create deck feature
           .assign(Cabin = lambda x : np.where(x.Cabin=='T',np.nan,x.Cabin))
           .assign(Deck = lambda x : x.Cabin.map(get_deck))
           #feature encoding
           .assign(IsMale = lambda x : np.where(x.Sex == 'male',1,0))
           .pipe(pd.get_dummies, columns= ['Deck','Pclass','Title','Fare_bin','Embarked','AgeState'])
           
           # add code to drop the unnecessary columns
           .drop(['Cabin','Name','Ticket', 'Parch','SibSp','Sex'],axis=1) # we dont need inplace here as we are using method chaining approach
           #reorder columns
           .pipe(reorder_columns)
          )
            
                        
    
def get_title(name):
    map_title = {'mr' : 'Mr',
                 'mrs' : 'Mrs',
                 'miss' : 'Miss',
                 'master': 'Master',
                 'don': 'Sir', 
                 'rev': 'Sir', 
                 'dr' : 'Officer', 
                 'mme' : 'Mrs',
                 'ms' : 'Mrs',
                 'major' : 'Officer',
                 'lady' : 'Lady', 
                 'sir' : 'Sir', 
                 'mlle' : 'Miss', 
                 'col' : 'Officer', 
                 'capt' : 'Officer', 
                 'the countess' : 'Lady',
                 'jonkheer' : 'Sir', 
                 'dona' : 'Lady'}
    first_name = name.split(',')[1]
    title=first_name.split('.')[0]
    title=title.strip().lower()
    return map_title[title]
           
def fill_missing_values(df):
    #embarked
    df.Embarked.fillna('C', inplace=True)
    #fare
    median_fare=df.loc[(df.Pclass==3) & (df.Embarked=='S'),'Fare'].median()
    df.Fare.fillna(median_fare,inplace=True)
    #age
    title_age_median=df.groupby('Title').Age.transform('median')
    df.Age.fillna(title_age_median, inplace=True)
    return df
    
def reorder_columns(df):
    columns=[col for col in df.columns if col !='Survived']
    columns=['Survived']+ columns
    df=df[columns]
    return df

def get_deck(cabin):
    return np.where(pd.notnull(cabin),str(cabin)[0].upper(),'Z')

def write_data(df):
    processed_data_path=os.path.join(os.path.pardir,'data','processed')
    write_train_pat=os.path.join(processed_data_path,'train.csv')
    write_test_pat=os.path.join(processed_data_path,'test.csv')
    # train data
    df.loc[df.Survived!=-888].to_csv(write_train_pat)
    columns_test=[col for col in df.columns if col !='Survived']
    df.loc[df.Survived==-888,columns_test].to_csv(write_test_pat)
    
# main function
if __name__ == '__main__':
    df=read_data()
    df=process_data(df)
    write_data(df)
    
