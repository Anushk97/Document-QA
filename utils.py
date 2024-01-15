from sentence_transformers import SentenceTransformer, util
import pinecone
import openai
import streamlit as st
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
import faiss

openai.api_key = "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"
#model = SentenceTransformer('multi-qa-distilbert-cos-v1')
#model = SentenceTransformer('all-mpnet-base-v2')

pinecone.init(api_key='09d08617-45d2-4ce8-b708-d8291d5570d6', environment='gcp-starter')
index = pinecone.Index('langchain-chatbot-v2')

if st.button("Reset"):
    index.delete(delete_all=True, namespace='langchain-chatbot-v2')

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')

index_dimension = model.config.hidden_size
index_f = faiss.IndexFlatIP(index_dimension)

def find_match(input):
    encoded_input = tokenizer(input, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    ##input_em = F.normalize(sentence_embeddings, p=2, dim=1).tolist()

    #sentence_embeddings = util.mean_pooling(model_output, encoded_input['attention_mask'])
    input_em = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1).cpu().numpy()

    # Perform similarity search using Faiss
    k = 2  # Top 2 matches
    index_f.add(input_em)
    distances, indices = index_f.search(input_em, k)

    # Retrieve metadata for the top matches (modify this based on your metadata structure)
    matches = [f"Match {i + 1}: {index_f.get_metadata(idx)}" for i, idx in enumerate(indices[0])]

    return matches

    #input_em = model.encode(input).tolist()
    ##result = index.query(input_em, top_k=2, includeMetadata=True)
    ##return result['matches'][0]['metadata']['text'] + result['matches'][1]['metadata']['text']


def query_refiner(query):
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
def query_refiner(query):
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    inputs = tokenizer(query, return_tensors="pt")
    
    outputs = model.generate(**inputs, max_new_tokens=256)

    refined_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return refined_text
'''
def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string
