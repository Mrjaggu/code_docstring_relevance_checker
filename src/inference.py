# importing libraries
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from lightgbm import LGBMClassifier
# from sklearn.metrics import f1_score
import pandas as pd
import joblib
import string
import re
import numpy as np
import random
import os
from config import SENTENCE_TRANSFORMER_PRE_TRAINED_MODEL_NAME,PREDICTION_MODEL_NAME
from sentence_transformers import SentenceTransformer
from code_extract import get_data
from collections import Counter
import math
import numpy
from stop_words import get_stop_words
import warnings
warnings.filterwarnings("ignore")

# intializing for parallel computing of embedding 
os.environ["TOKENIZERS_PARALLELISM"] = "True"

model_name = SENTENCE_TRANSFORMER_PRE_TRAINED_MODEL_NAME
model = SentenceTransformer(model_name)


# loading our fine-tuned lgbm model
lgb_model = joblib.load(PREDICTION_MODEL_NAME)   


STOP_WORDS = get_stop_words('english')

# STOP_WORDS = stopwords.words("english")

# some custom stopwords depending on the analysis what we captured
custom_stowords_list = ['def','self','returns','return',
                        'zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz',
                       'zzzzzzzzzz','zzzzzzzz','zzzzzzz','zzzzz',
                       'aaaabnzacyceaaaadaqabaaabaqddurxdpyhoquyodutxjduzqmjyjqjszt',
       'aaaabnzacyceaaaadaqabaaabaqdcgxehzf', 'aaaabnzacyceaaa',
       'aaaabnzacyce', 'aaaabnzacyc', 'aaaabnzacy',
       'aaaabnzackcmaaacbakdpkarimlm', 'aaaabbbccdaabbb', 'aaaabbbcca',
       'aaaabaaacaaadaaaeaaafaaagaaahaaaiaaajaaakaaalaaamaaanaaaoaaapaaaqaaaraaasaaataaa',
       'aaaaargh', 'aaaaabaaa', 'aaaaabaa',
       'aaaaaaeceeeeiiiidnoooooouuuuysaaaaaaaceeeeiiii', 'aaaaaabaedaa',
       'aaaaaaaarge', 'aaaaaaaalaaaaaaadwpwaaaaaaaaaa',
       'aaaaaaaalaaaaaaadwpqzmzmzmdk', 'aaaaaaaadjuqwsqicqdwlclimq',
       'aaaaaaaaaaagaagaaagaaa', 'aaaaaaaaaaaahaaaaaaaaa',
       'aaaaaaaaaaaaaaaaaadwpwaaaaaaapc',
       'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
       'aaaaaaaaaaaaaaaa', 'aaaaaaaaaaaaaaa', 'aaaaaaaaaaaaaa',
       'aaaaaaaaaaaaa', 'aaaaaaaaaaaa', 'aaaaaaaaaaa', 'aaaaaaaaaa',
       'aaaaaaaaa', 'aaaaaaaa', 'aaaaaaa', 'aaaaaa', 'aaaaa', 'aaaa',
       'aaa', 'aa']

STOP_WORDS.extend(custom_stowords_list)


def get_method_name(code:str)->str:
    """Extracting method name from method name as a feature"""
    
    return code.split(":")[0]


def preprocess(x):
    """removing stopwords and numeric from string """

    x = str(x).lower()
    y = []
    for zz in x.split():
        if zz.lower() not in STOP_WORDS or zz.lower() not in custom_stowords_list:
            y.append(zz)
    x = " ".join(y)        
    x = re.sub(r"([0-9]+)","", x)
    
    return x

def clean_data(lines):
    """" Method return cleaned string"""
    cleaned = []
    for line in lines.split("\n"):
        clean = re.sub(r"""
               [,.;@#?!&$()|^<='\\_`:>"%/{}*]+  # Accept one or more copies of punctuation
               \ *           # plus zero or more copies of a space,
               """,
               " ",          # and replace it with a single space
               line, flags=re.VERBOSE)
        # clean = re.sub(r"[^a-zA-Z0-9 ]+","",clean)
        #Manually handle cases not accepted by sub
        clean = clean.replace("[", "")
        clean = clean.replace("+", "")
        clean = clean.replace("]", "")
        clean = clean.replace("-", "")
        # tokenize on white space
        line = clean.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # store as string
        cleaned.append(' '.join(line))
    # remove empty strings
    return " ".join(cleaned)

def pre_processing_pipeline(df):
    """ pre-processing pipeline to clean the string """

    def convert_html_text(text):
        """ function convert http link to text"""

        if 'https://' in text:
            for pattern in string.punctuation:
                text = text.replace(pattern," ")

            return text
        else:
            return text
    
    
    def extract_features(df):
        # preprocessing each question
        print(f"[Info] Running cleaning pipeline!")
        
        df["code"] = df["code"].fillna("").apply(clean_data)
        df["docstring"] = df["docstring"].fillna("").apply(clean_data)
        df["method_feature"] = df["method_feature"].fillna("").apply(clean_data)
        
        print(f"[Info] Cleaning done!")
        
        return df
    
    ##conveting hyperlinks to string format
    df['docstring'] = df['docstring'].apply(
        convert_html_text
    )
    
    df = extract_features(df)
    
    print(f"[Info] Running stopwords removal pipeline!")
    
    df['code'] = df['code'].apply(preprocess)
    df['docstring'] = df['docstring'].apply(preprocess)
    df["method_feature"] = df["method_feature"].apply(preprocess)
    
    print(f"[Info] Stopwords removal complete!")
    
    return df
    

def counter_cosine_similarity(c1,c2):
    """method return cosine similarity between two strings"""
    
    terms = set(c1).union(c2)
    dotprod = sum(c1.get(k,0)*c2.get(k,0) for k in terms)
    magA = math.sqrt(sum(c1.get(k,0)**2 for k in terms))
    magB = math.sqrt(sum(c2.get(k,0)**2 for k in terms))
    
    try:
        return dotprod/(magA*magB)
    except:
        return 0
  

def get_sentence_embedding(list_of_sentences:pd.Series)->np:
    """ method return array of sentence embedding """
    
    feature_embedding = model.encode(list_of_sentences,
                                     show_progress_bar=True
                                    )
    
    return feature_embedding


def data_prepare(code_embedding_feature:numpy,
                doc_embedding_:numpy,
                code_embedding_:numpy,
                test_data:pd.DataFrame())->numpy:
    
    # concatenating our code, docsting, code method feature in one single numpy array
    train_c = np.concatenate([code_embedding_feature,
                              doc_embedding_,
                              code_embedding_],
                             axis=1)
    
    # using our features and converting it to numpy to add in embedding features
    features_array = test_data[['sim_score']].to_numpy()
    
    # concatenating similarity feature to embedding feature
    train_c_with_features = np.concatenate(
        [train_c,features_array],
        axis=1)
    
    return train_c_with_features
    

def get_prediction(train_c_with_features,test_data):
    #use loaded model to to get predictions
    pred_prob = lgb_model.predict(train_c_with_features)
    # probability
    pred_class = (pred_prob >=0.5)*1
    test_data['y_pred'] = pred_class

    if "id" not in test_data.columns:
        test_data['id'] = [x for x in range(len(test_data))]

    submission_file = test_data[['id','y_pred']] # only the id and y_pred to be part of output
    
    return submission_file


def batch_predict(test_data)->pd:

    test_data['method_feature'] = test_data['code'].apply(get_method_name)

    test_data = pre_processing_pipeline(test_data)


    test_data['sim_score'] = test_data.apply(lambda x: 
                                counter_cosine_similarity(Counter(x['code'].split(" ")),
                                                            Counter(x['docstring'].split(" "))),
                                axis=1
                                )

    code_embedding_ = get_sentence_embedding(test_data.code.values)
    code_embedding_feature = get_sentence_embedding(test_data.method_feature.values)
    doc_embedding_ = get_sentence_embedding(test_data.docstring.values)


    source = data_prepare(code_embedding_feature,doc_embedding_,code_embedding_,test_data)

    result_df = get_prediction(source,test_data)

    return result_df


def predict(code):

    data = get_data(code)
    result_df = batch_predict(data)

    return data,result_df

