import streamlit as st
from backend.model import predict_news

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection")

st.caption("Powered by PyTorch + TF-IDF")

st.divider()

news = st.text_area(
    "📝 Enter News Article",
    placeholder="Paste a news article here...",
    height=250
)

if st.button("🔍 Predict", use_container_width=True):

    if not news.strip():
        st.warning("Please enter a news article.")
        st.stop()

    with st.spinner("Analyzing..."):
        result = predict_news(news)

    st.divider()

    prediction = result["prediction"]
    confidence = result["confidence"]

    if prediction == "Real News":
        st.success(f"✅ {prediction}")
    else:
        st.error(f"❌ {prediction}")

    st.metric(
        label="Confidence",
        value=f"{confidence}%"
    )

    st.progress(confidence / 100)