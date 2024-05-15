from chain import chain5

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_core.tracers import LangChainTracer
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage
from langchain_openai import ChatOpenAI
from langsmith import Client
import streamlit as st
from streamlit_feedback import streamlit_feedback
import time
import os

# print(chain5.invoke("How is BTC performing"))

st.set_page_config(page_title="DeFi GPT", page_icon="🤑")
st.title("🤑 DeFi GPT")

os.environ["LANGCHAIN_PROJECT"] = "text-analytics-project"

# Customize if needed
client = Client()
ls_tracer = LangChainTracer(project_name=os.environ["LANGCHAIN_PROJECT"], client=client)
run_collector = RunCollectorCallbackHandler()
# cfg = RunnableConfig()
# cfg["callbacks"] = [ls_tracer, run_collector]
# cfg["configurable"] = {"session_id": "any"}


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")

reset_history = st.sidebar.button("Clear Chat History")

if len(msgs.messages) == 0 or reset_history:
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state["last_run"] = None


if "messages" not in st.session_state:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="How can I help you?")
    ]

# for msg in st.session_state.messages:
#     st.chat_message(msg.role).write(msg.content)

# If user inputs a new prompt, generate and draw a new response
if user_input := st.chat_input():
    user_input_str = str(user_input)
    st.session_state.messages.append(ChatMessage(role="user", content=user_input_str))
    # st.chat_message("user").write(user_input_str)
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        # chain_with_history = RunnableWithMessageHistory(
        #     chain5,
        #     lambda session_id: msgs,
        #     input_messages_key="question",
        #     history_messages_key="history",
        # )
        # history = msgs.messages  # Access the chat history directly
        # response = chain_with_history.invoke({"question": user_input_str, "history": history}, cfg)
        response = chain5.invoke({"question": user_input_str})
        st.session_state.messages.append(
            ChatMessage(role="assistant", content=response)
        )
    # st.session_state.last_run = run_collector.traced_runs[0].id

for msg in st.session_state.messages:
    if msg.role == "user":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)


@st.cache_data(ttl="2h", show_spinner=False)
def get_run_url(run_id):
    time.sleep(1)
    return client.read_run(run_id).url


# if st.session_state.get("last_run"):
#     run_url = get_run_url(st.session_state.last_run)
#     st.sidebar.markdown(f"[LangSmith Tracking 🛠️]({run_url})")
#     feedback = streamlit_feedback(
#         feedback_type="thumbs",
#         optional_text_label=None,
#         key=f"feedback_{st.session_state.last_run}",
#     )
#     if feedback:
#         scores = {"👍": 1, "👎": 0}
#         client.create_feedback(
#             st.session_state.last_run,
#             feedback["type"],
#             score=scores[feedback["score"]],
#             comment=feedback.get("text", None),
#         )
#         st.toast("Your feedback has been saved!", icon="📝")














