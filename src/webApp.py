import streamlit as st
from src.inference import predict
st.title("Code review Platform")

import streamlit as st

# Use the file_uploader widget to let the user upload a Python file
file = st.file_uploader("Choose a Python file to upload")

# If a file is uploaded, display its contents
if file is not None:
    f = file.read()
    code = f.decode("utf-8")
    # st.write(code)
    st.code(code)
    df_input,df_out = predict(code)

    st.write("Input pre-processed dataframe:")
    st.table(df_input)
    st.write("Running pipeline.....!!!")
    st.write("Prediction:")
    st.table(df_out)