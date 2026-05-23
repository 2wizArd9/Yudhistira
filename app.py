import streamlit as st

from utils import (
    load_artifacts,
    predict_with_explanations,
    run_model_comparison,
    render_prediction_explainability,
    render_visualizations,
)


st.set_page_config(
    page_title="AI-Powered Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme():
    st.markdown(
        """
        <style>
        :root{
            --bg:#070b14;
            --panel: rgba(17,26,46,.75);
            --panel2: rgba(17,26,46,.45);
            --text:#e7eefc;
            --muted:#a9b7da;
            --primary:#3b82f6;
            --primary2:#60a5fa;
            --border: rgba(96,165,250,.25);
        }

        body{ background: linear-gradient(180deg, #0b1220, #070b14) !important; color:var(--text); }
        .stApp { background: transparent; }

        /* Glass panels */
        .block-container{ padding-top: 1rem; }
        .stCard { background: var(--panel); border: 1px solid var(--border); border-radius: 16px; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: rgba(10,16,34,.85) !important;
            border-right: 1px solid rgba(96,165,250,.25);
        }

        /* Buttons */
        div.stButton > button{
            border-radius: 12px;
            background: linear-gradient(90deg, var(--primary), var(--primary2));
            border: 0px;
            color: white;
            font-weight: 700;
        }
        div.stButton > button:hover{ filter: brightness(1.05); }

        /* Progress / bars */
        .metric-card { background: var(--panel2); border:1px solid var(--border); border-radius: 14px; padding: 14px; }

        </style>
        """,
        unsafe_allow_html=True,
    )


apply_theme()


# Sidebar
with st.sidebar:
    st.markdown("# 📰 AI Fake News Detector")
    st.write("**NLP + ML** dashboard")

    st.markdown("---")

    st.subheader("Project Overview")
    st.write(
        "Detects whether a news headline/article is **REAL** or **FAKE** using TF-IDF + multiple Scikit-learn models."
    )

    st.subheader("Technologies")
    st.write("- Python\n- Streamlit\n- Scikit-learn\n- NLTK\n- TF-IDF\n- Pandas/NumPy\n")

    st.subheader("Model Info")
    with st.spinner("Loading model artifacts..."):
        artifacts = load_artifacts()
    st.write(f"Model file: **model.pkl**")
    st.write(f"Vectorizer file: **vectorizer.pkl**")

    st.subheader("Dataset")
    st.write("LIAR dataset (simplified to Real/Fake labels)")

    st.markdown("---")
    st.subheader("Developer")
    st.write("Built for portfolio/placements showcase ✨")


st.title("AI-Powered Fake News Detection System")

# Load artifacts already loaded in sidebar
model, vectorizer, extra = artifacts


st.markdown(
    """
    <div style="background: rgba(17,26,46,.45); border:1px solid rgba(96,165,250,.25); border-radius:16px; padding:16px;">
      <h3 style="margin:0">Try it now</h3>
      <p style="color: #a9b7da; margin-top:6px">Paste a headline or full article. The app will preprocess the text, generate TF‑IDF features, and predict the label with confidence + explainability.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.form("prediction_form"):
    text_input = st.text_area(
        "Paste news headline/article:",
        height=180,
        placeholder="Example: 'The government has approved a new vaccine...'",
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        example = st.button("Use example 🧠")
    with col2:
        clear = st.button("Clear ✨")
    with col3:
        run = st.form_submit_button("Predict 🚀")

    if example:
        text_input = "The rumor claims that a new law was passed secretly to eliminate a certain group."
        st.experimental_rerun()

    if clear:
        st.experimental_rerun()


if run:
    if not text_input.strip():
        st.error("Please enter some text to analyze.")
    else:
        with st.spinner("Preprocessing + predicting..."):
            result = predict_with_explanations(model, vectorizer, text_input)

        # Top metrics
        pred_label = result["label"]
        prob_fake = result["prob_fake"]
        prob_real = result["prob_real"]

        conf = result["confidence"]
        colA, colB = st.columns(2)
        with colA:
            st.markdown(
                f"""
                <div class='metric-card'>
                  <h4 style='margin-bottom:6px'>Prediction</h4>
                  <div style='font-size:28px;font-weight:800'>{pred_label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with colB:
            st.markdown(
                """
                <div class='metric-card'>
                  <h4 style='margin-bottom:8px'>Confidence</h4>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(max(conf / 100.0, 0.0), 1.0))
            st.write(f"**{conf:.1f}%** confident")


        st.success(f"The model predicts: **{pred_label}** with **{conf:.1f}%** confidence.")

        # Explainability section
        st.subheader("🔍 Why this prediction?")
        render_prediction_explainability(result)

        st.subheader("📊 Model visualizations")
        render_visualizations()

        # Model comparison
        st.subheader("🧪 Compare Models (Real/Fake)")
        comparison_df = run_model_comparison()
        st.dataframe(comparison_df, use_container_width=True)

        st.caption("Note: comparison metrics are computed from the training pipeline artifacts if available.")

