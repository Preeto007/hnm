# Databricks notebook source
# MAGIC %pip install langchain

# COMMAND ----------

# MAGIC %pip install openai tiktoken sentence-transformers faiss-cpu
# MAGIC

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

from ap.utils.azure.connector import AzureConnector
from langchain.schema.messages import SystemMessage, HumanMessage
import uuid
import pandas as pd
import streamlit as st
import numpy as np
from streamlit import set_page_config
def generate_marketing_mail():
    # Set page configuration
    set_page_config(page_title="Marketing Mail App")
    # Get Azure connectors
    client = AzureConnector.get_formrecognizer_client()
    client = AzureConnector.get_cognitive_search_index_client()
    model = AzureConnector.get_openai_chat_model()
    # Set params
    model.temperature = 0.5

    df = pd.read_parquet('/dbfs/mnt/hackathon/output/merged_inputs.parquet').head()
    df.head()
    # COMMAND ----------
    customers = list(set(df['CUSTOMER_NAME'].values))
    dict_replacement = {k:uuid.uuid4().__str__() for k in customers}
    df_00 = df.replace({'CUSTOMER_NAME':dict_replacement})
    for _, v in dict_replacement.items():
        df_local = df_00.query(f'CUSTOMER_NAME=="{v}"')
        crops = df_local['CROP_NAME'].values
        diseases = df_local['TARGET_NAME'].values
        products = df_local['PRODUCT_NAME'].values
        customer_name = df_local['CUSTOMER_NAME'].values
        customer_city = df_local['CITY'].values
        customer_state = df_local['STATE'].values
        template = '''Hello dear customer [customer name],
        As we noticed that you are growing crops like [crops] and we have detected that there is a risk for
        diseases like [diseases] to impact your crops, we thought that products like [products] could be interesting
        for you.
        We would be happy to send you more information to your addres, placed at the city [city name] at the state [state].
        Best regards,
        Company X
        '''
        messages = [
            SystemMessage(content="You are a sales representative from a chemical company selling treatments for crops."),
            SystemMessage(content="Your audience are the potential customers."),
            SystemMessage(content="Your task is to write a mail based on a template and filing missing information with data provided."),
            SystemMessage(content="The mail should be written in a formal but fiendly way. You are expected to vary the base template. Lenght of the mail should be about 200 words"),
            HumanMessage(content=f"As part of the data provided, my name as a company is Company X."),
            HumanMessage(content=f"As part of the data provided, my name as a sales rep is Mr X."),
            HumanMessage(content=f"As part of the data provided, these are the crops the customer grows: {', '.join(crops)}."),
            HumanMessage(content=f"As part of the data provided, these are the diseases that could impact these crops: {', '.join(diseases)}."),
            HumanMessage(content=f"As part of the data provided, these are the recommended products for the customer: {', '.join(products)}."),
            HumanMessage(content=f"As part of the data provided, customer name is {customer_name}, customer city is {customer_city} and customer state is {customer_state}."),
            HumanMessage(content=f"The template you could use as inspiration is as follows: {template}."),
            HumanMessage(content="Now you can generate the mail.")
        ]
        output = model(messages)
        if st.button("Send"):
            response = model.send_message(HumanMessage(output.content))
            st.text(f"Bot Response: {response.text}")


# COMMAND ----------

if __name__ == "__main__":
    generate_marketing_mail()

# COMMAND ----------

import os
os.environ["STREAMLIT_SERVER_PORT"] = "8888"
os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"

# COMMAND ----------

!streamlit run /databricks/python_shell/scripts/PythonShell.py --server.port 8888
