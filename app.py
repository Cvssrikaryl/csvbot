import os
from io import BytesIO
from PIL import Image
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import AzureOpenAI
from pandasai.responses.response_parser import ResponseParser

# Load environment variables
load_dotenv()

# Ensure temp folder exists
os.makedirs("temp", exist_ok=True)

# Azure OpenAI client setup
llm = AzureOpenAI(
    api_token=os.getenv("OPENAI_API_KEY"),
    azure_endpoint="https://hcmchaos.openai.azure.com/",
    api_version="2024-12-01-preview",
    deployment_name="gpt-4o",
    is_chat_model=True
)

# Custom ResponseParser for Streamlit UI
class StreamlitChatResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        query = st.session_state.get("current_query", "").lower()
        df_result = result["value"]
        code = result.get("code")

        if any(word in query for word in ["show", "display", "table", "dataframe", "list", "pivot"]):
            st.session_state["last_response"] = df_result
            st.session_state["last_type"] = "dataframe"
        elif isinstance(df_result, pd.DataFrame) and df_result.shape == (1, 1):
            st.session_state["last_response"] = str(df_result.iloc[0, 0])
            st.session_state["last_type"] = "text"
        else:
            st.session_state["last_response"] = "(Table generated but not shown. Use keywords like 'show table' to see it.)"
            st.session_state["last_type"] = "text"

        st.session_state["last_code"] = code

    def format_plot(self, result):
        query = st.session_state.get("current_query", "").lower()
        image_path = result["value"]
        code = result.get("code")

        if any(word in query for word in ["plot", "graph", "chart", "image", "visualize", "show"]):
            st.session_state["last_response"] = image_path
            st.session_state["last_type"] = "plot"
        else:
            st.session_state["last_response"] = "(Chart generated but not shown. Use 'show chart' or similar to display it.)"
            st.session_state["last_type"] = "text"

        st.session_state["last_code"] = code

    def format_other(self, result):
        st.session_state["last_response"] = result["value"]
        st.session_state["last_code"] = result.get("code")
        st.session_state["last_type"] = "text"

# Streamlit app config
st.set_page_config(page_title="CSV Chatbot", layout="centered")
st.title("FileQA bot")

st.markdown("""
<style>
    .chat-container {
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }
    .user-bubble, .bot-bubble {
        padding: 10px 15px;
        border-radius: 15px;
        max-width: 75%;
    }
    .user-bubble {
        background-color: #DCF8C6;
        align-self: flex-end;
    }
    .bot-bubble {
        background-color: #F1F0F0;
        align-self: flex-start;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📁 Upload CSV")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if st.button("🔁 Reset Chat"):
        for key in ["chat_history", "context", "last_code", "last_response", "last_type", "dataframe"]:
            st.session_state.pop(key, None)
        st.experimental_rerun()

# Load file
if uploaded_file:
    if "dataframe" not in st.session_state:
        st.session_state["dataframe"] = pd.read_csv(uploaded_file)
        st.session_state["chat_history"] = []
        st.session_state["context"] = ""
        st.session_state["last_code"] = ""
        st.session_state["last_response"] = ""
        st.session_state["last_type"] = "text"

    df = st.session_state["dataframe"]
    st.success("✅ CSV loaded successfully!")

    with st.expander("🔍 Preview Data", expanded=False):
        st.dataframe(df.head())

    chat_placeholder = st.container()
    with chat_placeholder:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for entry in st.session_state.chat_history:
            st.markdown(f'<div class="user-bubble">🧑‍💻 {entry["user"]}</div>', unsafe_allow_html=True)
            bot_type = entry.get("bot_type", "text")
            bot_response = entry["bot"]

            if bot_type == "text":
                st.markdown(f'<div class="bot-bubble">🤖 {bot_response}</div>', unsafe_allow_html=True)
            elif bot_type == "dataframe":
                st.markdown('<div class="bot-bubble">🤖 (DataFrame Response)</div>', unsafe_allow_html=True)
                with st.container():
                    st.dataframe(bot_response)
            elif bot_type == "plot":
                st.markdown('<div class="bot-bubble">🤖 (Image Response)</div>', unsafe_allow_html=True)
                with st.container():
                    if isinstance(bot_response, str) and os.path.exists(bot_response):
                        st.image(Image.open(bot_response))
                    elif isinstance(bot_response, Image.Image):
                        st.image(bot_response)
                    else:
                        st.markdown("⚠️ Unable to display image.")
            else:
                st.markdown(f'<div class="bot-bubble">🤖 (Unsupported type)</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    user_input = st.chat_input("Ask something about your data:")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state["current_query"] = user_input

        # Handle greetings or out-of-scope input
        greetings = ["hi", "hello", "hey", "hola"]
        out_of_scope = ["weather", "news", "joke", "who is", "what is your name"]

        user_input_lower = user_input.strip().lower()
        if any(user_input_lower.startswith(greet) for greet in greetings):
            bot_msg = "Hello! I am a CSV Question Answering bot. How can I assist you with your data?"
            st.session_state["chat_history"].append({"user": user_input, "bot": bot_msg, "bot_type": "text"})
            st.experimental_rerun()
        elif any(phrase in user_input_lower for phrase in out_of_scope):
            bot_msg = "I'm here to help with questions about your uploaded CSV data. Please ask something related to that."
            st.session_state["chat_history"].append({"user": user_input, "bot": bot_msg, "bot_type": "text"})
            st.experimental_rerun()

        query_engine = SmartDataframe(
            df,
            config={
                "llm": llm,
                "response_parser": StreamlitChatResponse,
                "custom_prompt": "This is cleaned financial data. Answer precisely based on it.",
                "save_charts": True,
                "save_path": "./temp",
                "verbose": True
            },
        )

        full_query = (
            st.session_state["context"]
            + f"\n\nUser: {user_input}"
            + (
                f"\n\n# Code used in prior answers:\n{st.session_state['last_code']}"
                if st.session_state.get("last_code") else ""
            )
        )

        try:
            query_engine.chat(full_query)
            result = st.session_state["last_response"]
            code = st.session_state["last_code"]
            bot_type = st.session_state["last_type"]

            if bot_type == "dataframe":
                bot_msg = "(DataFrame output shown below)"
                with st.chat_message("assistant"):
                    st.markdown(bot_msg)
                    st.dataframe(result)

            elif bot_type == "plot":
                bot_msg = "(Image output shown below)"
                with st.chat_message("assistant"):
                    st.markdown(bot_msg)
                    if isinstance(result, str) and os.path.exists(result):
                        st.image(Image.open(result))
                    elif isinstance(result, Image.Image):
                        st.image(result)

            else:
                bot_msg = result
                with st.chat_message("assistant"):
                    st.markdown(bot_msg)

            if code:
                st.session_state["context"] += f"\n\nAssistant: {bot_msg}\n\n# Code:\n{code}"
                bot_msg += f"\n\n🧠 Code Used:\n```python\n{code}\n```"
            else:
                st.session_state["context"] += f"\n\nAssistant: {bot_msg}"

            st.session_state["chat_history"].append({
                "user": user_input,
                "bot": result,
                "bot_type": bot_type
            })

            st.experimental_rerun()

        except Exception as e:
            st.error("❌ Error while processing your query.")
            st.exception(e)

    # 🔽 Persist download button after rerun if last result was a DataFrame
    if st.session_state.get("last_type") == "dataframe":
        result = st.session_state.get("last_response")
        if isinstance(result, pd.DataFrame):
            csv_bytes = result.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Last Result as CSV",
                data=csv_bytes,
                file_name="query_result.csv",
                mime="text/csv",
            )

else:
    st.info("👈 Upload a CSV file from the sidebar to begin.")
