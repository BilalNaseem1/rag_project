import warnings

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


from pymongo import MongoClient
from langchain.vectorstores import Pinecone as Pineconevectorstore
import time
import cohere
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import pandas as pd
from tqdm import tqdm
import pandas as pd
import pymongo
from pymongo import MongoClient
from googleapiclient.discovery import build
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv, dotenv_values
import os
import json
import time
from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
import anthropic
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
# from langchain-community.llms.octoai_endpoint import OctoAIEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from openai import OpenAI
import openai
import os
import together
from uuid import uuid4
from langchain_core.output_parsers import StrOutputParser
import ast
import itertools
import numpy as np
import logging
from typing import Any, Dict, List, Mapping, Optional
from langchain_community.embeddings import CohereEmbeddings
import cohere
from pydantic import Extra, Field, root_validator
import pymongo
# from langchain.storage import MongoDBStore
from langchain_community.vectorstores import Pinecone
# from langchain.retrievers.multi_vector import MultiVectorRetriever
from pymongo import MongoClient
import pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env

from IPython.display import display, Markdown

from anthropic import Anthropic
import json
import re
import textwrap
from pprint import pprint
from langchain_anthropic import ChatAnthropic
from tqdm import tqdm
import warnings

from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)


from langchain.load import dumps, loads
from langchain.chains.query_constructor.base import AttributeInfo
# from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

import os
import asyncio
from dotenv import find_dotenv, load_dotenv
import nest_asyncio

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
import json
from operator import itemgetter


from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRagRetriever, CohereRerank
from operator import itemgetter
from langchain_cohere import ChatCohere, CohereEmbeddings, CohereRagRetriever, CohereRerank
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
# from langchain.retrievers import ContextualCompressionRetriever
from langchain.text_splitter import CharacterTextSplitter
import cohere

from langchain.sql_database import SQLDatabase