import os
import io
import json
import psycopg2

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel
from PIL import Image

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
DB_URL = os.getenv("NEON_DATABASE_URL")

# Database Setup
def init_db():
    """
    Create the receipts table if it doesn't exist
    """
    if not DB_URL:
        return "Ensure the connection to the NEON database is set correctly in the .env file"
    
    with psycopg2.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS receipt_items (
                    id SERIAL PRIMARY KEY,
                    store_name VARCHAR(255),
                    item_name VARCHAR(255),
                    quantity INT,
                    price DECIMAL(10, 2)
                )
            """)
            conn.commit()

init_db()

# Pydantic Schemas for the AI model
class ReceiptItem(BaseModel):
    item_name: str
    quantity: int
    price: float

class ReceiptData(BaseModel):
    store_name: str
    items: list[ReceiptItem]

# Streamlit Interface
st.set_page_config(page_title="AI-Scanner", layout='wide')

client = None
if API_KEY:
    client = genai.Client(api_key = API_KEY)
else:
    st.error("GEMINI_API_KEY is not set.")

st.title("AI-Powered Receipt Scanner")
st.write("Upload a photo of a receipt or handwritten note and watch the AI organise it.")

left_col, right_col = st.columns([1, 1])


AI_PROMPT = """
Analyze this receipt or handwritten note. Extract the store name and a list of all purchased items.
Ensure quantities are integers and prices are floats. If a value is unreadable, estimate or use 0.
"""

FORMAT_TO_MIME = {"JPEG": "image/jpeg", "PNG": "image/png"}
image_bytes = None
mime_type = None

with left_col:
    st.subheader("Upload Photo")
    uploaded_file = st.file_uploader("Drop your receipt file here.", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Uploaded Document", width = 400)

        save_format = image.format if image.format in FORMAT_TO_MIME else "JPEG"
        mime_type = FORMAT_TO_MIME[save_format]

        img_byte_arr = io.BytesIO()
        if save_format == "JPEG" and image.mode in ["RGBA", "p"]:
            image = image.convert("RGB")
        image.save(img_byte_arr, format=save_format)
        image_bytes = img_byte_arr.getvalue()

with right_col:
    st.subheader("Organised Data")

    if uploaded_file is not None:
        if st.button("Let AI read it!", type='primary', disabled=client is None):
            with st.spinner("AI is thinking..."):
                try:
                    response = client.models.generate_content(
                        model = 'gemini-3.6-flash',
                        contents=[
                            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                            AI_PROMPT
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ReceiptData, 
                        )
                    )
                    
                    # Parse the JSON response
                    receipt_json = json.loads(response.text)
                    store = receipt_json.get("store_name", "Unknown Store")
                    items = receipt_json.get("items", [])
                    
                    st.success("Extraction Complete!")
                    
                    # Display the data as a clean dataframe
                    if items:
                        df = pd.DataFrame(items)
                        st.write(f"**Store:** {store}")
                        st.dataframe(df, use_container_width=True)
                    
                    # Insert into Neon PostgreSQL
                    if DB_URL and items:
                        with psycopg2.connect(DB_URL) as conn:
                            with conn.cursor() as cur:
                                for item in items:
                                    cur.execute(
                                        """INSERT INTO receipt_items 
                                           (store_name, item_name, quantity, price) 
                                           VALUES (%s, %s, %s, %s)""",
                                        (store, item["item_name"], item["quantity"], item["price"])
                                    )
                                conn.commit()
                        st.info("Data successfully saved to Neon database.")

                except Exception as e:
                    st.error(f"Something went wrong: {e}")
    else:
        st.info("Waiting for you to upload an image on the left!")

