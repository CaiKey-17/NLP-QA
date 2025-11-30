import streamlit as st
import torch

# ==== IMPORT MODEL LOADER CỦA BẠN ====
# copy lại 3 hàm load: load_phobert_model, load_vit5_model, load_qwen_model
from model import load_phobert_model, load_vit5_model, load_qwen_model

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# 📌 LOAD MODELS (lazy loading)
# ============================

@st.cache_resource
def load_models():
    return {
        "PhoBERT (Extractive)": load_phobert_model("D:\\NLP\\model\\phobert_qa_model_extend"),
        "ViT5 (Generative)":    load_vit5_model("D:\\NLP\\model\\vit5_qa_model_extend"),
        "Qwen 0.6B (Generative)": load_qwen_model("D:\\NLP\\model\\qwen3_qa_model_extend"),
    }

models = load_models()


# ============================
# 📌 APP UI
# ============================
st.set_page_config(page_title="Vietnamese QA Demo", layout="wide")

st.title("🇻🇳 Vietnamese QA Model Demo")
st.write("Demo thử nghiệm mô hình Hỏi–Đáp: PhoBERT, ViT5, Qwen")


# ====== SIDEBAR ======
st.sidebar.header("⚙️ Tuỳ chọn")
model_name = st.sidebar.selectbox(
    "Chọn mô hình",
    list(models.keys())
)

temperature = st.sidebar.slider("Nhiệt độ (dùng cho model generative)", 0.0, 1.5, 0.3)


# ====== INPUT AREA ======
st.subheader("📝 Nhập dữ liệu")

context = st.text_area("Ngữ cảnh:", height=200)
question = st.text_input("Câu hỏi:")

run = st.button("🚀 Run Model")


# ============================
# 📌 RUN MODEL
# ============================
if run:
    if not context.strip() or not question.strip():
        st.error("Vui lòng nhập đầy đủ Context và Question.")
    else:
        st.info(f"Đang chạy mô hình **{model_name}**…")

        model_fn = models[model_name]

        try:
            # tất cả các model bạn đã build đều tuân theo signature:
            # model_fn([contexts], [questions]) -> [answers]
            if model_name == "PhoBERT (Extractive)":
                answer = model_fn.predict([context], [question])[0]
            else:
                answer = model_fn([context], [question])[0]


            st.success("✨ Kết quả trả lời:")
            st.write(answer)

        except Exception as e:
            st.error(f"Lỗi khi chạy model: {str(e)}")

# ============================
# 📌 FOOTER
# ============================
st.markdown("---")
st.caption("Demo QA • Built for your final project • Streamlit + Huggingface models")
