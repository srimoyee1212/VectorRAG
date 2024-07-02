# Importing necessary libraries
from langchain import PromptTemplate
from langchain.memory import ConversationBufferMemory
import locale

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain.text_splitter import TokenTextSplitter
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Weaviate
import weaviate
import requests
import xml.etree.ElementTree as ET

# Setting preferred encoding to UTF-8
locale.getpreferredencoding = lambda: "UTF-8"

# Weaviate configuration
Weaviate_URL = "https://first-cluster-uirwx1yf.weaviate.network"
Weaviate_API_KEY = "3O2vYI0O4tOtFvZuiW8HVsDn9FgBiJBnh8EE"
client = weaviate.Client(url=Weaviate_URL, auth_client_secret=weaviate.AuthApiKey(Weaviate_API_KEY))

# Initializing HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", model_kwargs={"device": "cuda"})

# YouTube API configuration
URL = "https://www.youtube.com/feeds/videos.xml?channel_id=UCIC1L2vbbyotqEF0ZLhaOdw"
response = requests.get(URL)
xml_data = response.content
root = ET.fromstring(xml_data)
namespaces = {
    "atom": "http://www.w3.org/2005/Atom",
    "media": "http://search.yahoo.com/mrss/",
}

# Extracting YouTube video links
youtube_links = [link.get("href") for link in root.findall(".//atom:link[@rel='alternate']", namespaces)][1:]

# Checking Torch version and CUDA availability
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(torch.version.cuda)

# Loading YouTube video transcripts
from langchain.document_loaders import YoutubeLoader
all_docs = []
for link in youtube_links:
    loader = YoutubeLoader.from_youtube_url(link)
    docs = loader.load()
    all_docs.extend(docs)

# Splitting text into tokens for processing
text_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=0)
split_docs = text_splitter.split_documents(all_docs)

# Creating and populating Weaviate vector database
vector_db = Weaviate.from_documents(split_docs, embeddings, client=client, by_text=False)

# Performing similarity search on vector database
vector_db.similarity_search("How to protect against attacks on critical infrastructure", k=3)

# Loading and initializing quantized language model
model_name = "anakin87/zephyr-7b-alpha-sharded"
def load_quantized_model(model_name: str):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    return model

# Initializing tokenizer for language model
def initialize_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_token_type_ids=False)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer

tokenizer = initialize_tokenizer(model_name)

# Loading quantized model
model = load_quantized_model(model_name)

# Setting stop token ids for model
stop_token_ids = [0]

# Initializing text generation pipeline
pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    use_cache=True,
    device_map="auto",
    max_length=2048,
    do_sample=True,
    top_k=5,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# Initializing HuggingFace pipeline for text generation
llm = HuggingFacePipeline(pipeline=pipeline)

# Template for response generation
template = """
Use the following context (delimited by ) and the chat history (delimited by ) to answer the question:
------

{context}

------

{history}

------
{question}
Answer:
"""
prompt = PromptTemplate(
    input_variables=["history", "context", "question"],
    template=template,
)

# Initializing retrieval-QA pipeline
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=vector_db.as_retriever(), verbose=True,
    chain_type_kwargs={
        "verbose": True,
        "prompt": prompt,
        "memory": ConversationBufferMemory(
            memory_key="history",
            input_key="question"),
    }
)

# Running retrieval-QA pipeline with a sample query
response = qa_chain.run("What are some recent cybersecurity attacks?")
print(response)
