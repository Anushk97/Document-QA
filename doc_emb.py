from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone

directory = 'data'

def load_docs(pdf):
  loader = PyPDFLoader(pdf)
  documents = loader.load_and_split()
  return documents[0]

def split_docs(documents,chunk_size=5000,chunk_overlap=50):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs


#print(len(docs))

embeddings = SentenceTransformerEmbeddings(model_name="multi-qa-distilbert-cos-v1")

# query_result = embeddings.embed_query("otot")

# initialize pinecone
pinecone.init(
    api_key="09d08617-45d2-4ce8-b708-d8291d5570d6",  # find at app.pinecone.io
    environment="gcp-starter"  # next to api key in console
)

index_name = "langchain-chatbot-v2"

#index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

'''
def get_similiar_docs(query,k=1,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs
'''
#query = "what is the net profit?"
#similar_docs = get_similiar_docs(query)
#similar_docs
