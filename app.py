import streamlit as st
import pandas as pd

from intent_module import detect_intent
from faq_module import faq_module
from recommendation_module import recommend_products
from product_module import product_question
from review_module import review_module


# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="E-Commerce AI Assistant",
    page_icon="🛍️",
    layout="wide"
)


# ------------------------------------------------
# LOAD DATA
# ------------------------------------------------

products = pd.read_csv("data/products.csv")
faqs = pd.read_csv("data/faqs.csv")
reviews = pd.read_csv("data/reviews.csv")


# ------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: gray;
    font-size: 18px;
    margin-bottom: 30px;
}

.product-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #ddd;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------
# HEADER
# ------------------------------------------------

st.markdown(
    '<div class="main-title">🛍️ E-Commerce AI Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask about products, orders, recommendations and reviews</div>',
    unsafe_allow_html=True
)


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

with st.sidebar:

    st.header("📊 Dataset Information")

    st.metric(
        "Products",
        len(products)
    )

    st.metric(
        "FAQs",
        len(faqs)
    )

    st.metric(
        "Reviews",
        len(reviews)
    )

    st.divider()

    st.subheader("🤖 Capabilities")

    st.write("✅ FAQ Search")
    st.write("✅ Product Recommendation")
    st.write("✅ Product Q&A")
    st.write("✅ Review Analysis")
    st.write("✅ Sentiment Analysis")
    st.write("✅ Ollama LLM")
    st.write("✅ Intent Detection")


# ------------------------------------------------
# CHAT HISTORY
# ------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


# Display previous messages

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ------------------------------------------------
# USER INPUT
# ------------------------------------------------

query = st.chat_input(
    "Ask me anything about our products..."
)


if query:

    # Display user message

    st.session_state.messages.append(
        {
            "role": "user",
            "content": query
        }
    )

    with st.chat_message("user"):
        st.markdown(query)


    # ------------------------------------------------
    # PROCESS QUERY
    # ------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                # Detect intent
                intent = detect_intent(query)

                st.caption(
                    f"Detected Intent: {intent}"
                )


                # ----------------------------------------
                # FAQ
                # ----------------------------------------

                if intent == "FAQ":

                    response = faq_module(query)

                    if response:

                        st.markdown(response)

                    else:

                        st.warning(
                            "I couldn't find a relevant FAQ."
                        )


                # ----------------------------------------
                # RECOMMENDATION
                # ----------------------------------------

                elif intent == "RECOMMENDATION":

                    response = recommend_products(query)

                    st.markdown(response)


                # ----------------------------------------
                # PRODUCT
                # ----------------------------------------

                elif intent == "PRODUCT":

                    response = product_question(query)

                    st.markdown(response)


                # ----------------------------------------
                # REVIEW
                # ----------------------------------------

                elif intent == "REVIEW":

                    # Try to identify product
                    # from the query

                    matched_products = products[
                        products["ProductName"]
                        .str.lower()
                        .apply(
                            lambda x: x in query.lower()
                        )
                    ]

                    if len(matched_products) > 0:

                        product_name = matched_products.iloc[0][
                            "ProductName"
                        ]

                        st.info(
                            f"Analyzing reviews for: "
                            f"{product_name}"
                        )

                        response = review_module(product_name)
                        if response:
                            st.markdown(response)

                    else:

                        st.warning(
                            "Please mention the exact "
                            "product name."
                        )


            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )


    # Save assistant response
    # for chat history

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Response generated successfully."
        }
    )