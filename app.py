import streamlit as st
from transformers import pipeline

# 1. Setup Page Config
st.set_page_config(page_title="Letterboxd Sentiment Detective", page_icon="🎬")

# 2. Load the AI Model (Cached so it's fast)
@st.cache_resource
def load_model():
    # This model is specifically fine-tuned for movie reviews
    return pipeline("sentiment-analysis", model="sarahai/movie-sentiment-analysis")

classifier = load_model()

# 3. Streamlit UI
st.title("🎬 Letterboxd Sentiment Detective")
st.markdown("Paste a Letterboxd review below to see if the critic *actually* enjoyed the film.")

# User Input
review_text = st.text_area("Enter review text:", placeholder="e.g., 'A visual masterpiece, though the pacing felt like a slow crawl through molasses.'", height=150)

if st.button("Analyze Sentiment"):
    if review_text.strip():
        with st.spinner("Analyzing..."):
            # Run the AI
            result = classifier(review_text)[0]
            label = result['label']
            score = result['score']

            # Display Results
            if label.lower() == "positive":
                st.success(f"### Result: POSITIVE 😊")
                st.write(f"Confidence: {score:.2%}")
            else:
                st.error(f"### Result: NEGATIVE 🍿 (Dropped popcorn)")
                st.write(f"Confidence: {score:.2%}")
    else:
        st.warning("Please enter some text first!")

st.divider()
st.caption("Built with Streamlit + Hugging Face Transformers")
