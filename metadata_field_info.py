from langchain.load import dumps, loads
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI

metadata_field_info = [
    AttributeInfo(
        name="Author",
        description="Author of the Video",
        type="string",
    ),
    AttributeInfo(
        name="year",
        description="Year on which the video was published",
        type="integer",
    ),
    AttributeInfo(
        name="month",
        description="Month on which the video was published",
        type="string",
    ),
    AttributeInfo(
        name="date",
        description="Date on which the video was published",
        type="integer",
    ),
    AttributeInfo(
        name="text",
        description="Summary of the video",
        type="string",
    ),
    AttributeInfo(
        name="title", description="Title of the video", type="string"
    ),
    AttributeInfo(
        name="url", description="url of the source", type="string"
    ),
]
document_content_description = "Summaries of youtube videos on cryptocurrencies. Do not use the word 'contains' or 'contain' or 'like' as filters."