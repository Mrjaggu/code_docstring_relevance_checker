from typing import List
import pandas as pd
import re

def get_code_as_string(code:str)->str:

    def_line_no = []
    # with open(filepath) as f:
    #     lines = f.readlines()
    lines = code.split("\n")
    for idx, x in enumerate(lines):
        
        if "def" in x:
            def_line_no.append(idx)

    code_ =[]
    for l in range(len(def_line_no)):
        if l+1 < len(def_line_no):
            code_.append(" ".join(lines[def_line_no[l]:def_line_no[l+1]]))

        else:
            code_.append(" ".join(lines[def_line_no[l]:]))

    return code_

def prepare_data(def_line_number):

    pattern = r'"""(.*?)"""'
    final_processed_data = []
    for text in def_line_number:

        text = text.replace("\n","")
        
        # print(text)
        result = re.findall(pattern, text)
        # print(result)
        if len(result)!=0:

            text = re.sub(result[0],"",text)
            # print(text)
            final_processed_data.append([text,result[0]])

    
    return final_processed_data


# def get_data(filepath = r'C:\AJAY\HACKATHON\test.py'):

#     def_line_number = get_code_as_string(filepath)


#     df = pd.DataFrame(prepare_data(def_line_number),columns=["code","docstring"])
#     df.to_csv('sample.csv',index=False)

#     return df

def get_data(code):

    def_line_number = get_code_as_string(code)


    df = pd.DataFrame(prepare_data(def_line_number),columns=["code","docstring"])
    df.to_csv('sample.csv',index=False)

    return df
