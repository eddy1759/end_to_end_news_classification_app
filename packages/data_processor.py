import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 

def prepare_data(path_to_data):
    """
    Args:
        path_to_data ([str]): [the path to the dataset]

    return:
        - dictionary containing the vectorized features and labels.
    """

    #Read data from the path
    df = pd.read_csv(path_to_data)
    df['label_id'] = df['Category'].factorize()[0]
    label_id = df[['Category', 'label_id']].drop_duplicates().sort_values('label_id')
    id_to_category = dict(label_id[['label_id', 'Category']].values)

    X = df.Text
    y = df.label_id

    return {'features':X,'labels':y}, id_to_category


def create_train_test_split(X,y,test_size,random_state):
    """[split the dataset into training and test set]

    Args:
        X ([array]): [the features for training]
        y ([array]): [the labels]
        test_size ([float]): [the percentage of testing size]
        random_state ([int]): [random state]      
    """

    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', ngram_range=(1, 2), stop_words='english')
    X = tfidf.fit_transform(X).toarray()

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,
                                    random_state=random_state)

    return{'x_train': X_train,'x_test': X_test,
            'y_train':y_train,'y_test':y_test
            },tfidf


            