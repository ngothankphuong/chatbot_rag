import os
import cohere
import numpy as np
import json
import re
import glob
import requests
import faiss
# import seqtoseq
# import Semantic router
# from semantic_router.router import semantic_router

from flask import Flask, request, jsonify, session, redirect,url_for, flash
from langchain.callbacks import StdOutCallbackHandler
from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains import StuffDocumentsChain
from langchain import hub
from langchain.globals import set_verbose, set_debug
from langchain_core.prompts.chat import (ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate,)
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from pdfminer.high_level import extract_text
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage

api_key = ""
os.environ["COHERE_API_KEY"] = api_key

llm = ChatCohere(model="command-r-plus", temperature=0)

# URL API của Rasa (thay đổi địa chỉ IP và cổng nếu khác)
RASA_API_URL = "http://127.0.0.1:5005/webhooks/rest/webhook"

current_directory = os.getcwd()
co = cohere.Client(api_key)
base_embedd = CohereEmbeddings(cohere_api_key=api_key, model="embed-multilingual-v3.0")

app = Flask(__name__)
# app.secret_key = 'your_secret_key'

path_default_document = os.path.join(current_directory, 'data\\documents_short.txt')

path_for_FAISS = os.path.join(current_directory, 'FAISS_DB')

datenow = datetime.now()
formatted_time = datenow.strftime('%d-%m-%Y-%H-%M-%S')

folder_split_doc = os.path.join(current_directory, 'split_docs')

path_for_split_file = os.path.join(folder_split_doc, f'{formatted_time}.txt')


UPLOAD_FOLDER = os.path.join(current_directory, 'file_upload')