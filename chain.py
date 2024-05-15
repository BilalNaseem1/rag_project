from importing_modules import *

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
import streamlit as st


#--- Setting up LangSmith ---#

nest_asyncio.apply()
load_dotenv(find_dotenv())
os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "text-analytics-project"
load_dotenv()
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
MONGO_CLIENT = os.environ.get('MONGO_CLIENT')
COHERE_API_KEY = os.environ.get('COHERE_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_INDEX_NAME = os.environ.get('PINECONE_INDEX_NAME')


#--- Setting up MongoDb and Pinecone ---#

# import certifi
# ca = certifi.where()

# client = pymongo.MongoClient(
# "mongodb+srv://bilalnaseem:PIFks9OWxElsOjl3@textproject.aqw6crm.mongodb.net/xyzdb?retryWrites=true&w=majority", tlsCAFile=ca)

client = MongoClient(MONGO_CLIENT)
db = client.TextProject
collection = db.transcripts
embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)
vectorstore = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
co = cohere.Client(COHERE_API_KEY)


from pinecone import ServerlessSpec, PodSpec, Pinecone
spec = ServerlessSpec(cloud='aws', region='us-west-2')

index_name = "masterindex2"
# configuring client
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)

# Embedding

def embed_text(text):
    result = co.embed(texts=[text], model="embed-english-v3.0", input_type="search_document")
    time.sleep(0.2)  # Sleep for 0.2 seconds after each API call
    return result.embeddings[0]

# def perform_search(user_query):
#     user_query = str(user_query)
#     user_embedding = embed_text(user_query)
#     results = index.query(vector=user_embedding, top_k=3, include_metadata=True)
#     document_ids = [result.id for result in results.matches]
#     documents = collection.find({"_id": {"$in": document_ids}})
#     document_texts = [doc["Transcript"] for doc in documents]
#     return document_texts

llm = ChatAnthropic(temperature=0, max_tokens=4000, model_name="claude-3-haiku-20240307", anthropic_api_key=ANTHROPIC_API_KEY)


# Generate alternative questions
generate_queries_template = """You are an AI language model assistant. Your task is to generate three 
different versions of the given user question to retrieve relevant documents from a vector 
database. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. Original question: {question}"""

def filter_docs(docs):
    return [q for q in docs[1:] if q != '']


prompt_perspectives = ChatPromptTemplate.from_template(generate_queries_template)
generate_queries = (
    prompt_perspectives 
    | llm
    | StrOutputParser() 
    | (lambda x: x.split("\n"))
    | RunnableLambda(filter_docs)
)

# # Query Structuring

# from metadata_field_info import metadata_field_info
# from metadata_field_info import document_content_description

# from attribute_info import attribute_info
# import datetime
# from typing import Literal, Optional, Tuple
# from langchain_core.pydantic_v1 import BaseModel, Field


# from langchain.chains.query_constructor.base import (
#     get_query_constructor_prompt,
#     load_query_constructor_runnable,
#     StructuredQueryOutputParser
# )
# query_constructor_chain = (
#     generate_queries
#     | selfq_retriever.map() 
#     | RunnableLambda(get_unique_union)
# )

# from langchain.retrievers.self_query.pinecone import PineconeTranslator

# vectorstore = Pineconevectorstore.from_existing_index(index_name="benjamin-cowen-summ3", embedding=embeddings)

# selfq_retriever = SelfQueryRetriever.from_llm(
#     llm,
#     vectorstore,
#     document_content_description,
#     metadata_field_info,
#     PineconeTranslator(),
#     fix_invalid=True,
#     verbose=True,
#     top_k = 3
# )
######################################################################################

def get_unique_union(documents: dict):
    """ Unique union of retrieved docs """
    # Flatten list of lists, and convert each Document to string
    flattened_docs = [dumps(doc) for sublist in documents['context'] for doc in sublist]
    # Get unique documents
    unique_docs = list(set(flattened_docs))
    # Return
    return {"context": [loads(doc) for doc in unique_docs], "question": documents['question']}



def perform_search(user_query):
    user_query = str(user_query)
    user_embedding = embed_text(user_query)
    results = index.query(vector=user_embedding, top_k=3, include_metadata=True)
    document_ids = [result.id for result in results.matches]
    documents = collection.find({"_id": {"$in": document_ids}})
    document_texts = [doc["Transcript"] for doc in documents]
    return document_texts


# def perform_search2(queries):
#     all_results = []
#     for q in queries:
#         x = perform_search(q)
#         all_results.append(x)
#     return all_results

# def qs_prompt(question):
#     qs_prompt = question + "\n" + "Answer based solely on the below video transcripts of Benjamin Cowen, a YouTuber known for his technical analysis of the crypto market.\n"
#     return qs_prompt

# def retrieval_chain(question):
#     queries = generate_queries.invoke({"question": question})
#     documents = [perform_search(query) for query in queries]
#     unique_docs = get_unique_union(documents)

#     # Rerank using Cohere
#     co = cohere.Client("kyIT3CZ30dCn6RJpIkmHB5EXnv53O92LAKIr7h66")
#     reranked_docs = co.rerank(
#         query=str(question),
#         documents=unique_docs,
#         top_n = 3,
#         model="rerank-english-v3.0"
#     )
#     return reranked_docs

def search_documents(queries: dict):
    documents = [perform_search(queries['context']) for query in queries]
    return {"context": documents, "question": queries['question']}

co = cohere.Client("kyIT3CZ30dCn6RJpIkmHB5EXnv53O92LAKIr7h66")

# def rerank_wrapper(input_dict: dict):
#     return rerank_documents(input_dict["context"], input_dict["question"])

chain2 = (
    RunnableParallel({"context": generate_queries, "question": RunnablePassthrough()})
    | RunnableLambda(search_documents)
    | RunnableLambda(get_unique_union)
    # | RunnableLambda(rerank_documents)
)

def chainrank(dct):
    # x = chain2.invoke(question)
    reranked_results = co.rerank(
        query=dct['question']['question'],  ##################
        documents=dct['context'],
        top_n=3,
        model="rerank-english-v3.0", 
        return_documents=True
    )
    context = ""
    for idx, r in enumerate(reranked_results.results):
        context += r.document.text
        context += '\n'

    return context

from langchain_core.prompts import ChatMessagePromptTemplate, MessagesPlaceholder

# history_placeholder = MessagesPlaceholder("history")
generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert financial research assistant. Answer the user's queries given the context, and also chat history if it's related to the user's question."),
        # MessagesPlaceholder(variable_name="history"),
        ("human", """
        Please answer the following query based on the provided context. Please cite your resources at the end of your responses.

        Query: {question}
        -----
        <context>
        {context}
        </context>
        """)
    ]
)


chain3 = (
    chain2
    | RunnableParallel({"context": RunnableLambda(chainrank), "question": RunnablePassthrough()})
    | generation_prompt
    | llm
)
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=1000)  

# chain3 = ConversationChain(
#     llm=llm,
#     memory=memory,
#     prompt=generation_prompt,
#     input_key="question",
#     output_key="answer",
#     verbose=True,
# )

# x = chain3.invoke("How is btc performing?")
# print(x)




# #################################################### SQL Chain ####################################################

project = "taproject-422806"
dataset = "coindataset"
service_account_path = './gbqkey2.json'
url = f'bigquery://{project}/{dataset}?credentials_path={service_account_path}'
db = SQLDatabase.from_uri(url)

from langchain.chains import create_sql_query_chain


BIGQUERY_PROMPT_TEXT = '''You are a bigquery expert. Given an input question, first create a syntactically correct bigquery query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per bigquery. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Only use the following tables:
{table_info}

In big query my dataset name is 'coindataset' and table name is 'coins' so you should query from coindataset.coins
Strictly Do not enclose column names in inverted commas as it does not conform to bigquery syntax. Strictly follow the syntax given in the Example below.
Return only the SQL Query and nothing else.

<example>
SELECT price
FROM coindataset.coins
WHERE name = 'Polkadot'
LIMIT 1;
</example>

Question: {input}'''

BIGQUERY_PROMPT = PromptTemplate(input_variables=['input', 'table_info', 'top_k'], template=BIGQUERY_PROMPT_TEXT)


sql_query_chain = create_sql_query_chain(llm, db, prompt=BIGQUERY_PROMPT)

# def replace_from_clause(answer):
#     return answer.replace("", )



def run_query_and_interpret(question):
    response = sql_query_chain.invoke({"question": question})
    response = response.replace('"', '')
    # sql_query_pattern = r"SQLQuery: (.*)"
    # match = re.search(sql_query_pattern, response, re.DOTALL)
    # if match:
    #     sql_query = match.group(1)
    # else:
    #     print("SQL query not found in the response.")
    #     return

    # Execute the SQL query and interpret the result
    from google.cloud import bigquery
    from google.oauth2 import service_account
    import pandas as pd

    credentials = service_account.Credentials.from_service_account_file(service_account_path)
    client = bigquery.Client(project=project, credentials=credentials)
    query_job = client.query(str(response))
    rows = query_job.result()
    results = str(query_job.to_dataframe())

    return results



sql_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert financial research assistant who is an expert in SQL."),
        ("human", """
        The following sql query:
        Query: {question}
        -----
        gave the following output
        -----
        <context>
        {context}
        </context>
        <Instructions>
        Interpret the results in a short format
        <examples>
        Question: What is the current price of BTC
        Output: The price of BTC is $50,000
        Question: What is the Price to TVL of BTC
        Output: The price to TVL ratio of BTC is 0.07
        </examples>
        </Instructions>
        """)
    ]
)

sql_chain = (
    RunnableParallel({"context": run_query_and_interpret, "question": RunnablePassthrough()})
    | sql_prompt
    | llm
)


#################################################### Routing ####################################################
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_cohere import ChatCohere

# Data model
class Pinecone_vectorsores(BaseModel):
    """
    A vectorstore containing documents, news, articles, video transcripts related to cryptocurrencies.
    """

    query: str = Field(description="The query to use when searching the internet.")


class tabular_data(BaseModel):
    """
    A Table containing price, TVL (Total volume locked), total supply and circulating supply of cryptocurrencies. Use the vectorstore for questions on these topics.
    """

    query: str = Field(description="The query to use when searching the vectorstore.")


# Preamble
preamble = """You are an expert at routing a user question to a vectorstore or an sql database
The vectorstore contains documents, news, articles, video transcripts related to cryptocurrencies.
The SQL database contains price, TVL (Total volume locked), total supply and circulating supply of cryptocurrencies.
Use the SQL database only when the user asks about price, TVL (Total volume locked), total supply and circulating supply of coins. Otherwise, use vectorstore."""


# LLM with tool use and preamble
cohere_llm = ChatCohere(model="command-r", temperature=0)
structured_llm_router = cohere_llm.bind_tools(
    tools=[Pinecone_vectorsores, tabular_data], preamble=preamble
)

# Prompt
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}"),
    ]
)

# question_router = route_prompt | structured_llm_router


def handle_query(response):
    tool_calls = response['context'].response_metadata["tool_calls"]
    if tool_calls:
        tool_name = tool_calls[0]["function"]["name"]
        if tool_name == "Pinecone_vectorsores":
            result = chain3.invoke(response['question'])
            return result
        elif tool_name == "tabular_data":
            result = sql_chain.invoke(response['question']['question']) #################
            return result
        else:
            fallback_response = cohere_llm.generate([("human", response['question'])])
            return fallback_response
    else:
        fallback_response = cohere_llm.generate([("human", response['question'])])
        return fallback_response

chain4 = (
    route_prompt
    | structured_llm_router
)

chain5 = (
    RunnableParallel({"context": chain4, "question": RunnablePassthrough()})
    | RunnableLambda(handle_query)
    | StrOutputParser()
)
# x = chain5.invoke("What is the price to tvl ratio of polkadot?")
# x = chain5.invoke("What is the price to tvl ratio of ethereum?")
# x = chain5.invoke("How is BTC going to perform after halving?")

# x = chain5.invoke({
#     "question": "Is it better to hold cardano or btc in bear market?",
#     "chat_history": 
# })
# print(x)
