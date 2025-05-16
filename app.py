import os
from io import BytesIO
from PIL import Image
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import AzureOpenAI
from pandasai.responses.response_parser import ResponseParser

# Load environment variables
load_dotenv()

# Ensure ./temp folder exists
os.makedirs("temp", exist_ok=True)

# Azure OpenAI LLM setup
llm = AzureOpenAI(
    api_token=os.getenv("OPENAI_API_KEY"),
    azure_endpoint="https://hcmchaos.openai.azure.com/",
    api_version="2024-12-01-preview",
    deployment_name="gpt-4o",
    is_chat_model=True
)

# Custom response parser for Streamlit
class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        df_result = result["value"]
        if isinstance(df_result, pd.DataFrame):
            st.session_state["query_result_df"] = df_result
            st.session_state["chat_history"][-1]["bot_type"] = "dataframe"
            st.session_state["chat_history"][-1]["bot"] = df_result
        else:
            st.warning("Expected a DataFrame, got something else.")

    def format_plot(self, result):
        try:
            image_path = result["value"]
            if isinstance(image_path, str) and os.path.exists(image_path):
                st.session_state["chat_history"][-1]["bot_type"] = "plot"
                st.session_state["chat_history"][-1]["bot"] = image_path
            else:
                st.warning("Plot image path is invalid.")
        except Exception as e:
            st.warning("Error displaying plot.")
            st.exception(e)

    def format_other(self, result):
        value = result["value"]
        st.session_state["chat_history"][-1]["bot_type"] = "text"
        st.session_state["chat_history"][-1]["bot"] = value

# === Streamlit UI Setup ===
st.set_page_config(page_title="Conversational CSV Chatbot", layout="centered")

# Styling for ChatGPT-style bubbles
st.markdown("""
<style>
    .block-container { max-width: 850px !important; margin: auto; }
    .user-bubble {
        background-color: #DCF8C6;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 10px 0px;
        max-width: 75%;
        align-self: flex-end;
    }
    .bot-bubble {
        background-color: #F1F0F0;
        border-radius: 15px;
        padding: 10px 15px;
        margin: 10px 0px;
        max-width: 75%;
        align-self: flex-start;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
    }
</style>
""", unsafe_allow_html=True)

st.title("Conversational CSV Chatbot")

# Sidebar file upload
with st.sidebar:
    st.header("📁 Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if st.button("🔁 New Chat"):
        st.session_state.pop("chat_history", None)
        st.session_state.pop("query_result_df", None)
        st.experimental_rerun()

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV uploaded successfully!")

        with st.expander("🔍 Preview Data", expanded=False):
            st.dataframe(df.head())

        # Initialize session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query_engine = SmartDataframe(
            df,
            config={
                "llm": llm,
                "response_parser": StreamlitResponse,
                "custom_prompt": "This is cleaned financial data. Answer precisely based on it.",
                "save_charts": True,
                "save_path": "./temp",
            },
        )

        # Display chat history
        chat_placeholder = st.container()
        with chat_placeholder:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            for entry in st.session_state.chat_history:
                st.markdown(
                    f'<div class="user-bubble">🧑‍💻 {entry["user"]}</div>',
                    unsafe_allow_html=True
                )
                bot_type = entry.get("bot_type", "text")
                bot_response = entry["bot"]

                if bot_type == "text":
                    st.markdown(f'<div class="bot-bubble">🤖 {bot_response}</div>', unsafe_allow_html=True)
                elif bot_type == "dataframe":
                    st.markdown('<div class="bot-bubble">🤖 (DataFrame Response)</div>', unsafe_allow_html=True)
                    st.dataframe(bot_response)
                elif bot_type == "plot":
                    st.markdown('<div class="bot-bubble">🤖 (Plot Response)</div>', unsafe_allow_html=True)
                    if os.path.exists(bot_response):
                        img = Image.open(bot_response)
                        st.image(img)
                else:
                    st.markdown('<div class="bot-bubble">🤖 (Unsupported response type)</div>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Input box for user
        user_input = st.chat_input("Ask something about your data:")

        if user_input:
            st.session_state.chat_history.append({"user": user_input, "bot": None, "bot_type": None})
            try:
                query_engine.chat(user_input)
                st.experimental_rerun()
            except Exception as e:
                st.error("❌ Error while processing your query.")
                st.exception(e)

        # Download result
        if "query_result_df" in st.session_state:
            csv_bytes = st.session_state["query_result_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Result as CSV",
                data=csv_bytes,
                file_name="query_result.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error("❌ Error reading the CSV file.")
        st.exception(e)
else:
    st.info("👈 Upload a CSV from the sidebar to begin.")
