
import os
import io
import psycopg2

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

# Load Environment Variables
load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
DB_URL = os.getenv("NEON_DATABASE_URL")

# Database Setup
def init_db():
    """Create the receipt_items table if it does not exist."""

    if not DB_URL:
        return False

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

    return True

# Data Validation
class ReceiptItem(BaseModel):
    item_name: str
    quantity: int
    price: float


class ReceiptData(BaseModel):
    is_receipt: bool
    store_name: str = ""
    items: list[ReceiptItem] = Field(default_factory=list)


AI_PROMPT = """
Your first task is to determine whether the uploaded image
is a valid receipt or purchase record.

Accept:
- Printed shopping receipts.
- Handwritten purchase notes listing purchased items.
- Other clear purchase records containing purchased items
  and their prices or quantities.

Reject:
- Ordinary photographs.
- Job advertisements.
- Promotional messages and marketing flyers.
- Posters and unrelated documents.
- Images that do not contain a purchase record.

If the image is NOT a valid receipt or purchase record:
- Set is_receipt to false.
- Set store_name to an empty string.
- Return an empty items list.
- Do not extract or invent purchase information.

If the image IS a valid receipt or purchase record:
- Set is_receipt to true.
- Extract the store or seller name when available.
- Extract all identifiable purchased items.
- Ensure quantities are integers and prices are numbers.
- If a quantity or price is unreadable, use 0.
- Use an empty store name if the seller cannot be identified.

Return only the information required by the JSON schema.
Do not include additional explanations.
"""


st.set_page_config(
    page_title="RecscanAI",
    layout="wide"
)

client = None

if API_KEY:
    client = genai.Client(api_key=API_KEY)
else:
    st.error(
        "GEMINI_API_KEY is not set. "
        "Please configure it in your environment."
    )

if not DB_URL:
    st.warning(
        "NEON_DATABASE_URL is not set. "
        "Extracted data will not be saved."
    )
else:
    try:
        init_db()
    except Exception:
        st.error(
            "Database initialisation failed. "
            "Check your Neon database connection."
        )


st.title("RecscanAI")
st.write(
    "Upload a receipt or handwritten purchase note. "
    "AI will extract the purchase information and organise it."
)

left_col, right_col = st.columns([1, 1])

uploaded_file = None
image_bytes = None
mime_type = None

with left_col:
    st.subheader("Upload Receipt")

    uploaded_file = st.file_uploader(
        "Upload a receipt image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image.load()

            st.image(
                image,
                caption="Uploaded Document",
                width=400
            )

            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")

            image_buffer = io.BytesIO()
            image.save(image_buffer, format="JPEG")

            image_bytes = image_buffer.getvalue()
            mime_type = "image/jpeg"

        except (UnidentifiedImageError, OSError, ValueError):
            st.error(
                "The uploaded file is not a valid image. "
                "Please upload a valid receipt image."
            )
            uploaded_file = None


with right_col:
    st.subheader("Organised Data")

    if uploaded_file is None:
        st.info("Upload a receipt image to get started.")

    elif image_bytes is not None:
        if st.button(
            "Read my receipt!",
            type="primary",
            disabled=client is None
        ):
            with st.spinner("Checking and reading the image..."):
                try:
                    response = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=[
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type=mime_type
                            ),
                            AI_PROMPT
                        ],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=ReceiptData
                        )
                    )

                    if not response.text:
                        st.error(
                            "The AI could not process this image. "
                            "Please try again."
                        )
                        st.stop()

                    receipt = ReceiptData.model_validate_json(
                        response.text
                    )

                    if not receipt.is_receipt:
                        st.error(
                            "This isn't a receipt. Upload a valid receipt."
                        )
                        st.stop()
                        break

                    if not receipt.items:
                        st.warning(
                            "No purchased items could be extracted. "
                            "Please upload a clearer receipt."
                        )
                        st.stop()

                    store = receipt.store_name or "Unknown Store"

                    df = pd.DataFrame(
                        [
                            item.model_dump()
                            for item in receipt.items
                        ]
                    )

                    st.success("Receipt extraction complete!")
                    st.write(f"**Store:** {store}")

                    st.dataframe(
                        df,
                        use_container_width=True,
                        hide_index=True
                    )

                    if DB_URL:
                        with psycopg2.connect(DB_URL) as conn:
                            with conn.cursor() as cur:
                                for item in receipt.items:
                                    cur.execute(
                                        """
                                        INSERT INTO receipt_items
                                            (
                                                store_name,
                                                item_name,
                                                quantity,
                                                price
                                            )
                                        VALUES (%s, %s, %s, %s)
                                        """,
                                        (
                                            store,
                                            item.item_name,
                                            item.quantity,
                                            item.price
                                        )
                                    )

                        st.success(
                            "Saved!"
                        )
                    else:
                        st.warning(
                            "Extraction succeeded, but the data "
                            "was not saved because the database "
                            "connection is not configured."
                        )

                except errors.APIError as e:
                    if getattr(e, "code", None) == 503:
                        st.error(
                            "The model is temporarily unavailable. "
                            "Try again in a few minutes."
                        )
                    else:
                        st.error(
                            f"Gemini API error: {e}"
                        )

                except psycopg2.Error:
                    st.error(
                        "A database error occurred. "
                        "The receipt data could not be saved."
                    )

                except Exception:
                    st.error(
                        "Something went wrong while processing "
                        "the receipt. Please try again."
                    )
