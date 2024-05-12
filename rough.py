from importing_modules import *

from langchain.memory import ConversationSummaryMemory, ChatMessageHistory
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory
import streamlit as st

llm = ChatAnthropic(temperature=0, max_tokens=4000, model_name="claude-3-haiku-20240307", anthropic_api_key="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")

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=200)
st.title("DeFi GPT")
user_input = st.text_input("Enter your message here")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history=[]
else:
    for message in st.session_state.chat_history:
        memory.save_context(
            {'input': message['human']},
            {'output': message['AI']}
            )

prompt_template = PromptTemplate(
    input_variables=['history', 'input'],
    template="""
    You are a conversational bot who answers the users queries.

    conversation history:
    {history}

    human: {input}
    AI:      
    """
)


conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    memory=memory,
    verbose=True
)

if user_input:
    response = conversation_chain(user_input)
    message = {
        'human': user_input,
        'AI': response['text']
    }
    st.session_state.chat_history.append(message)
    st.write(response['text'])
    with st.expander(label='Chat History', expanded=False):
        st.write(st.session_state.chat_history)