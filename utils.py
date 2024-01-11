from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st
from transformers import pipeline
import torch

openai.api_key = "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"
model = SentenceTransformer('multi-qa-distilbert-cos-v1')

pinecone.init(api_key='09d08617-45d2-4ce8-b708-d8291d5570d6', environment='gcp-starter')
index = pinecone.Index('langchain-chatbot-v2')


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em, top_k=10, includeMetadata=True)
    return result['matches'][0]['metadata']['text'] + result['matches'][1]['metadata']['text']

'''
def query_refiner(conversation, query):
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Given the following user query, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base. \n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].text
'''

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

def query_refiner(conversation, query):
    response = pipe(prompt = "Given the following user query, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base", 
                     max_new_tokens=256, do_sample=True, temperature=0.7, top_k=5, top_p=0.95)
    
    return response.choices[0].text


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string
