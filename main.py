from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from io import StringIO
from langchain.llms import CTransformers
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

import streamlit as st
from streamlit_chat import message
from utils import *
from huggingface_hub import hf_hub_download
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from transformers import AutoModelForCausalLM
from langchain.llms import HuggingFaceHub
from huggingface_hub import InferenceClient
from transformers import AutoModelForQuestionAnswering
from doc_emb import *

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

st.subheader("LLM Chatbot")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["How can I assist you?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

model_name_or_path = "TheBloke/Llama-2-13B-chat-GGML"
model_basename = "llama-2-13b-chat.ggmlv3.q5_1.bin" # the model is in bin format
model_path = hf_hub_download(repo_id=model_name_or_path, filename=model_basename)

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
'''
llm = CTransformers(
 model= model_path,
 model_type = 'llama',
 max_tokens=512,
 temperature = 0.5,
 n_gpu_layers=40,
 n_batch=512,
 top_p=0.95,
 repeat_penalty=1.2,
 top_k=1,
 callback_manager=callback_manager,
 verbose=True)
'''

# Make sure the model path is correct for your system!
'''
llm = LlamaCpp(
    model_path='llama-2-7b-chat.Q4_K_M.gguf',
    temperature=0.75,
    max_tokens=512,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
'''
### OPEN AI
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")


### HUGGINGFACE

repo_id = "google/flan-t5-base"

llm = HuggingFaceHub(
    repo_id=repo_id, model_kwargs={"temperature": 0.1, "max_length": 500})


#llm = AutoModelForQuestionAnswering.from_pretrained(repo_id)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=1,return_messages=True)

uploaded_file = st.file_uploader("Choose a file", type='pdf')
if uploaded_file:
   filename = st.text_input('input here', key='input_1')
   temp_file = filename
   with open(temp_file, "wb") as file:
       file.write(uploaded_file.getvalue())
       file_name = uploaded_file.name

   loader = PyPDFLoader(temp_file)
   pages = loader.load_and_split()
   docs = split_docs(pages)
#loader = PyPDFLoader(tmp_location)
#pages = loader.load_and_split()
   pinecone.init(
       api_key="09d08617-45d2-4ce8-b708-d8291d5570d6",  # find at app.pinecone.io
       environment="gcp-starter"  # next to api key in console
   )

   embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
   index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

#index_name = "langchain-chatbot"
#index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

prompt = st.text_input('prompt template', '')

system_msg_template = SystemMessagePromptTemplate.from_template(template=prompt)

human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")

prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)


# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()


with textcontainer:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            conversation_string = get_conversation_string()
            # st.code(conversation_string)
            refined_query = query_refiner(conversation_string, query)
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            # print(context)
            response = conversation.predict(input=f"Context:\n {context} \n\n Query:\n{query}")
        st.session_state.requests.append(query)
        st.session_state.responses.append(response)

with response_container:
    if st.session_state['responses']:

        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i],key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True,key=str(i)+ '_user')