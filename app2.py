import streamlit as st
#from langchain_ollama import Ollama
from langchain_community.llms.ollama import Ollama
  # ✅ Updated import
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings  # ✅ Updated import
from langchain.docstore.document import Document
import tempfile
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

# ------------------ Helper: Extract Text with OCR (PyMuPDF + Tesseract) ------------------
def extract_text_and_ocr(uploaded_files):
    all_text = ""
    for file in uploaded_files:
        file_type = file.type

        if file_type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            doc = fitz.open(tmp_path)
            for page_num, page in enumerate(doc, start=1):
                page_text = page.get_text()
                if not page_text.strip():
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    page_text = pytesseract.image_to_string(img)
                all_text += f"\n--- Page {page_num} ---\n{page_text.strip()}"
            doc.close()

        elif file_type.startswith("image/"):
            img = Image.open(file)
            text = pytesseract.image_to_string(img)
            all_text += f"\n--- Image: {file.name} ---\n{text.strip()}"

        else:
            all_text += f"\nUnsupported file type: {file.name}\n"
    return all_text.strip()

# ------------------ Helper: Vector Store Creation ------------------
def create_vector_store(text):
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# ------------------ Helper: ArXiv Related Papers ------------------
def find_related_papers(query):
    try:
        url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=3"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "xml")
        entries = soup.find_all("entry")
        return [f"[{e.title.text.strip()}]({e.id.text.strip()})" for e in entries]
    except:
        return ["❌ Error fetching related papers."]

# ------------------ Streamlit App ------------------
st.set_page_config(page_title="AutoResearcher AI", layout="wide")
st.title("🧪 AutoResearcher AI: AI-Augmented Research Companion")
st.caption("Advanced Research Paper Explainer with AI Agents, Visual Insights & Web Explorer")

uploaded_files = st.sidebar.file_uploader("📄 Upload Research PDFs or Images", type=["pdf", "png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    with st.spinner("📚 Reading & Processing documents..."):
        full_text = extract_text_and_ocr(uploaded_files)
        vector_store = create_vector_store(full_text)
        retriever = vector_store.as_retriever()
        llm = Ollama(model="tinyllama")  # ✅ Updated usage
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

    st.subheader("🧠 Ask anything about the uploaded papers")
    user_input = st.chat_input("Ask a question, e.g., What methodology is used?")

    if "history" not in st.session_state:
        st.session_state.history = []

    if user_input:
        with st.spinner("🤔 Thinking..."):
            try:
                answer = qa_chain.invoke({"input": user_input})  # ✅ Replaced .run()
                st.session_state.history.append((user_input, answer["result"]))
            except Exception as e:
                st.error(f"Error: {str(e)}")

    for q, a in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    # Related Papers Suggestion
    st.divider()
    st.subheader("🔎 Discover Related Research (arXiv)")
    rel_query = st.text_input("Search related topics (e.g., Explainable AI for healthcare):")
    if st.button("Fetch arXiv Links") and rel_query:
        results = find_related_papers(rel_query)
        for r in results:
            st.markdown("- " + r)

    # Blog Generation Prompt
    st.divider()
    st.subheader("✍️ AI-Generated Blog")
    if st.button("Generate Blog from Papers"):
        prompt = (
            "Generate a 500-word blog post summarizing the main contributions, methods, and results "
            "of the research papers. Include impact and future work."
        )
        try:
            blog = qa_chain.invoke({"input": prompt})
            st.text_area("Generated Blog:", blog["result"], height=300)
        except Exception as e:
            st.error(f"Blog generation failed: {e}")

    # Insight Visualizer
    st.divider()
    st.subheader("📊 Topic Frequency Chart (NER/Keyword Simulation)")
    with st.expander("Show Topic Insights"):
        keywords = [w for w in full_text.split() if len(w) > 5]
        freq = {}
        for word in keywords:
            freq[word] = freq.get(word, 0) + 1
        top_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        if top_words:
            labels, counts = zip(*top_words)
            fig, ax = plt.subplots()
            sns.barplot(x=list(counts), y=list(labels), ax=ax)
            st.pyplot(fig)
        else:
            st.info("No meaningful keywords found.")

    # Future Suggestion
    st.divider()
    st.info("📌 Coming Soon: Collaborative multi-agent research assistant, auto-reference builder, and real-time voice query support.")

else:
    st.info("⬅️ Upload one or more research PDFs or images to begin.")
    st.sidebar.info("Upload PDFs or images to extract text and ask questions about the content.")
    st.sidebar.info("Explore related research papers and generate blog posts based on the content.")
    st.sidebar.info("Visualize topic frequency and gain insights from your research documents.")
    st.sidebar.info("Stay tuned for more features like collaborative research and voice queries.")
