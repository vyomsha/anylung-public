import streamlit as st
from PIL import Image
import os

from inference.infer import load_model, run_inference

st.set_page_config(page_title="AnyLung Research Demo", layout="centered")

# ----------------------------
# Session-state gates
# ----------------------------
if "accepted_terms" not in st.session_state:
    st.session_state.accepted_terms = False
if "accepted_warning" not in st.session_state:
    st.session_state.accepted_warning = False

# ----------------------------
# Page 1: Terms & Conditions
# ----------------------------
if not st.session_state.accepted_terms:
    st.title("Terms & Conditions of Use")
    st.write(
        "This website provides access to an **experimental AI research demonstration**.\n\n"
        "- It is **not** a medical device.\n"
        "- It is **not** a diagnostic tool.\n"
        "- It is **not** intended for clinical use.\n\n"
        "The output may be inaccurate or misleading and must **not** be used to make medical decisions."
    )
    st.write("By continuing, you agree to use this system for **research/educational purposes only**.")
    if st.button("I Accept the Terms & Conditions"):
        st.session_state.accepted_terms = True
        st.rerun()
    st.stop()

# ----------------------------
# Page 2: Research Warning (per session)
# ----------------------------
if not st.session_state.accepted_warning:
    st.title("Important Research Notice")
    st.warning(
        "This is a research and educational demonstration. "
        "It does **not** indicate the presence or absence of disease."
    )
    st.info("If you have health concerns, consult a qualified medical professional.")
    if st.button("Continue to Demo"):
        st.session_state.accepted_warning = True
        st.rerun()
    st.stop()

# ----------------------------
# Load model once (CPU)
# ----------------------------
@st.cache_resource
def get_model():
    model_path = os.path.join("model", "anylung_model.pth")
    return load_model(model_path)

model = get_model()

# ----------------------------
# Demo UI
# ----------------------------
st.title("AnyLung Research Demo")
st.caption("Experimental AI behavior exploration under variable image quality.")

st.write("Upload an image (PNG/JPG). Use public/example images only.")
uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_container_width=True)

    st.subheader("Before running inference")
    confirm = st.checkbox("I understand this output is experimental and not medically reliable.")
    if not confirm:
        st.stop()

    if st.button("Run Research Inference"):
        with st.spinner("Running model..."):
            result = run_inference(model, img)

        st.subheader("Model Output (Research Only)")
        st.write(f"**Model confidence score:** `{result.score:.3f}`")
        st.write(f"**Model label:** `{result.label}`")
        st.info(result.notes)

        st.markdown("---")
        st.caption("No images are stored by this demo (design intent).")
else:
    st.markdown("Tip: add 2–3 example images in your repo and provide download links in README.")
