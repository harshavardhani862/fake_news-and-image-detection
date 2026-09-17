import streamlit as st
from PIL import Image
from text_model import predict_text
from image_model import predict_image


# ─────────────────────────────────────────────
# Load calibrated threshold automatically
# ─────────────────────────────────────────────
def load_threshold():
    try:
        with open("models/best_threshold.txt", "r") as f:
            return float(f.read().strip())
    except:
        return 0.40   # ✅ FIXED (was too high before)


# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News and Image Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News And Image Detection")
st.markdown("Detect whether news is **Real** or **Fake** using Text, Image, or Both.")
st.info("Tip: Longer text + clear images = better accuracy")
st.divider()


# ─────────────────────────────────────────────
# Result display function
# ─────────────────────────────────────────────
def show_result(label, confidence, probs):
    if label == "FAKE":
        st.error(f"🔴 Prediction: **{label}**")
    else:
        st.success(f"🟢 Prediction: **{label}**")

    col1, col2 = st.columns(2)
    col1.metric("🟢 REAL Probability", f"{probs[0]*100:.2f}%")
    col2.metric("🔴 FAKE Probability", f"{probs[1]*100:.2f}%")

    st.markdown(f"**Confidence: {confidence:.2f}%**")
    st.progress(min(int(confidence), 100))   # ✅ FIX: prevent overflow


# ─────────────────────────────────────────────
# Sidebar threshold control (FIXED RANGE)
# ─────────────────────────────────────────────
default_threshold = load_threshold()

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.30,     # ✅ LOWERED (important fix)
    max_value=0.70,     # ✅ balanced range
    value=default_threshold,
    step=0.01,
    help="Probability above which content is classified as FAKE."
)

st.sidebar.markdown(f"**Current Threshold: {threshold:.2f}**")


# ─────────────────────────────────────────────
# Fusion logic (IMPROVED)
# ─────────────────────────────────────────────
def fuse_predictions(t_probs, i_probs, threshold):
    """
    Combine text + image predictions.

    KEY FIX: When image fake probability is high (>= threshold),
    image gets more weight so a low text score cannot cancel it out.
    """

    img_fake  = i_probs[1]
    text_fake = t_probs[1]

    # Dynamic weighting — trust whichever model is more confident
    if img_fake >= threshold:
        image_weight = 0.70
        text_weight  = 0.30
    elif text_fake >= threshold:
        text_weight  = 0.70
        image_weight = 0.30
    else:
        text_weight  = 0.50
        image_weight = 0.50

    final_fake_prob = text_weight * text_fake + image_weight * img_fake
    final_real_prob = 1 - final_fake_prob

    # If EITHER model is very confident about FAKE (>= 0.75), override to FAKE
    # Prevents a low text score from cancelling a clearly fake image signal
    if img_fake >= 0.75 or text_fake >= 0.75:
        final_fake_prob = max(final_fake_prob, threshold + 0.01)
        final_real_prob = 1 - final_fake_prob

    if final_fake_prob > threshold:
        label      = "FAKE"
        confidence = final_fake_prob * 100
    else:
        label      = "REAL"
        confidence = final_real_prob * 100

    return label, confidence, [final_real_prob, final_fake_prob]


# ─────────────────────────────────────────────
# Input selection
# ─────────────────────────────────────────────
option = st.radio(
    "Select Input Type",
    ["📝 Text", "🖼️ Image", "🔀 Both"],
    horizontal=True
)


# ─────────────────────────────────────────────
# TEXT MODE
# ─────────────────────────────────────────────
if option == "📝 Text":

    text = st.text_area(
        "Enter news text here",
        height=180,
        placeholder="Paste or type news content..."
    )

    if st.button("🔍 Predict", type="primary", use_container_width=True):

        if text.strip():
            try:
                with st.spinner("Analyzing text..."):
                    label, confidence, probs = predict_text(text, threshold=threshold)

                st.divider()
                show_result(label, confidence, probs)

            except Exception as e:
                st.error(f"Error during text prediction: {e}")

        else:
            st.warning("⚠️ Please enter some text before predicting.")


# ─────────────────────────────────────────────
# IMAGE MODE
# ─────────────────────────────────────────────
elif option == "🖼️ Image":

    file = st.file_uploader("Upload a news image", type=["jpg", "jpeg", "png"])

    if file:
        try:
            img = Image.open(file).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)

            if st.button("🔍 Predict", type="primary", use_container_width=True):

                with st.spinner("Analyzing image..."):
                    label, confidence, probs = predict_image(img, threshold=threshold)

                st.divider()
                show_result(label, confidence, probs)

        except Exception as e:
            st.error(f"Error processing image: {e}")


# ─────────────────────────────────────────────
# BOTH MODE (IMPROVED)
# ─────────────────────────────────────────────
else:

    text = st.text_area(
        "Enter news text here",
        height=150,
        placeholder="Paste or type news content..."
    )

    file = st.file_uploader("Upload a news image", type=["jpg", "jpeg", "png"])

    if st.button("🔍 Predict Both", type="primary", use_container_width=True):

        if not text.strip():
            st.warning("⚠️ Please enter some text.")

        elif not file:
            st.warning("⚠️ Please upload an image.")

        else:
            try:
                img = Image.open(file).convert("RGB")

                with st.spinner("Analyzing text and image..."):
                    t_label, t_conf, t_probs = predict_text(text, threshold=threshold)
                    i_label, i_conf, i_probs = predict_image(img, threshold=threshold)

                    final_label, final_conf, final_probs = fuse_predictions(
                        t_probs, i_probs, threshold
                    )

                st.divider()

                # Individual results
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 📝 Text Result")
                    show_result(t_label, t_conf, t_probs)

                with col2:
                    st.markdown("### 🖼️ Image Result")
                    st.image(img, use_column_width=True)
                    show_result(i_label, i_conf, i_probs)

                # Final decision
                st.divider()
                st.markdown("## 🧠 Final Combined Decision")
                show_result(final_label, final_conf, final_probs)

            except Exception as e:
                st.error(f"Error during combined prediction: {e}")