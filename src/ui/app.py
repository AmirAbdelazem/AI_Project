import streamlit as st

class PredictorUI:
    def __init__(self, predictor):
        self.predictor = predictor

    def run(self):
        st.set_page_config(page_title="Sherlock Predictor", page_icon="🔍")
        st.title("🔍 Sherlock Holmes Next-Word Predictor")
        
        st.sidebar.header("Model Settings")
        top_k = st.sidebar.slider("Number of Suggestions", 1, 10, 3)
        
        user_input = st.text_input("Start typing...", placeholder="It has long been an axiom of mine...")

        if user_input:
            # The UI asks the Predictor for results
            predictions = self.predictor.predict_next(user_input, top_k)
            
            if predictions:
                st.subheader("Suggestions")
                # Display results as clickable buttons
                cols = st.columns(len(predictions))
                for i, word in enumerate(predictions):
                    with cols[i]:
                        if st.button(word):
                            st.success(f"Selected: {word}")
            else:
                st.info("The model needs more context or hasn't seen these words.")