import os
from io import BytesIO
from PIL import Image
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai.llm import AzureOpenAI
from pandasai.responses.response_parser import ResponseParser

# Load environment variables from .env file
load_dotenv()

# Ensure ./temp folder exists
os.makedirs("temp", exist_ok=True)

# Initialize Azure OpenAI
llm = AzureOpenAI(
    api_token=os.getenv("OPENAI_API_KEY"),
    azure_endpoint="https://hcmchaos.openai.azure.com/",
    api_version="2024-12-01-preview",
    deployment_name="gpt-4o",
    is_chat_model=True
)

# class StreamlitCallback(BaseCallback):
#     def __init__(self, container) -> None:
#         """Initialize callback handler."""
#         self.container = container

#     def on_code(self, response: str):
#         self.container.code(response)

# Custom response parser for Streamlit
class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)
        self.last_dataframe = None

    def format_dataframe(self, result):
        df_result = result["value"]
        st.dataframe(df_result)
        st.session_state["query_result_df"] = df_result

    def format_plot(self, result):
        try:
            image_path = result["value"]
            img = Image.open(image_path)
            st.image(img)
        except Exception as e:
            st.warning(f"Could not load image from path: {image_path}")
            st.exception(e)

    def format_other(self, result):
        st.write(result["value"])

# Streamlit UI
st.set_page_config(page_title="CSV Chatbot with PandasAI")
st.title("CSV Data Chatbot")

uploaded_file = st.file_uploader("📁 Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    with st.expander("🔍 Dataframe Preview"):
        st.dataframe(df.tail(5))

    query = st.text_area("🗨️ Ask a question about your data:")

    if query:
        query_engine = SmartDataframe(
            df,
            config={
                "llm": llm,
                "response_parser": StreamlitResponse,
                "save_charts": True,
                "save_path": "./temp",  # Portable path
            },
        )

        query_engine.chat(query)

        if "query_result_df" in st.session_state:
            csv_bytes = st.session_state["query_result_df"].to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Result as CSV",
                data=csv_bytes,
                file_name="query_result.csv",
                mime="text/csv",
            )