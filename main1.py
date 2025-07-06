# Final Updated Multi-PDF Chatbot with Structured Output Fix

import streamlit as st
import os
import time
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Streamlit setup
st.set_page_config(page_title="PDF-BOT Q&A", page_icon="📄", layout="centered")
st.markdown("<h2 style='text-align: center;'>✨ PDF BASED-AI BOT (Analytics + Q&A) ✨</h2>", unsafe_allow_html=True)
st.caption("Upload multiple PDFs and ask questions, summarize, extract points or compare them! ")

# Welcome message
if "welcomed" not in st.session_state:
    st.session_state.welcomed = True
    st.info(" Upload PDFs and ask anything like 'Summarize this document' or 'Compare the PDFs'.")

# Upload PDFs
pdf_files = st.file_uploader("📄 Upload your PDFs", type=["pdf"], accept_multiple_files=True)

# Small talk handler
def handle_small_talk(query):
    query = query.lower().strip()
    if query in ["hi", "hello", "hey"]:
        return "👋 Hello! How can I help you today?"
    elif query in ["bye", "goodbye", "exit"]:
        return "👋 Goodbye! See you next time."
    elif "thank" in query:
        return "You're welcome! 😊"
    elif "who are you" in query:
        return "I'm your AI assistant. Upload PDFs and ask me anything!"
    return None

# Detect task type from query
def detect_task(user_input):
    task_keywords = {
        "summarize": "summarize",
        "summary": "summarize",
        "bullet": "bullet_points",
        "key points": "bullet_points",
        "compare": "compare",
        "main topic": "main_topic",
    }
    for keyword, task in task_keywords.items():
        if keyword in user_input.lower():
            return task
    return "qa"

# Prompt handlers with Markdown-friendly formatting
def handle_summarization(text, llm):
    prompt = (
        "Summarize the document using short paragraphs under proper **headings** in markdown format."
        f"\n\n{text[:5000]}"
    )
    return llm.invoke(prompt)

def handle_bullet_points(text, llm):
    prompt = (
        "List the most important key points from the following content using markdown bullet points "
        "with **section headers** if applicable:\n\n"
        f"{text[:5000]}"
    )
    return llm.invoke(prompt)

def handle_comparison(text1, text2, llm):
    prompt = (
        "Compare the following two documents in a clear markdown table and bullet format. "
        "Use **headings**, bullet points, and **tables** to structure the output.\n\n"
        f"Document 1:\n{text1[:5000]}\n\nDocument 2:\n{text2[:5000]}"
    )
    return llm.invoke(prompt)

# Process PDFs
if pdf_files:
    with st.spinner("🔍 Processing your documents..."):
        all_docs = []
        for pdf in pdf_files:
            with open(pdf.name, "wb") as f:
                f.write(pdf.read())

            loader = PyPDFLoader(pdf.name)
            pages = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = splitter.split_documents(pages)

            for doc in docs:
                doc.metadata["source"] = pdf.name
            all_docs.extend(docs)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectorstore = FAISS.from_documents(all_docs, embeddings)

        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

        if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory(
                memory_key="chat_history", return_messages=True, output_key="answer"
            )

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=st.session_state.memory,
            return_source_documents=True,
            output_key="answer"
        )

        if "chat" not in st.session_state:
            st.session_state.chat = []

        query = st.chat_input("Ask a question, or say 'summarize this', 'list key points', etc...")

        if query:
            st.session_state.chat.append(("user", query))
            task_type = detect_task(query)

            with st.spinner("🤖 Thinking..."):
                if task_type == "summarize":
                    content = " ".join([doc.page_content for doc in all_docs])
                    answer = handle_summarization(content, llm)

                elif task_type == "bullet_points":
                    content = " ".join([doc.page_content for doc in all_docs])
                    answer = handle_bullet_points(content, llm)

                elif task_type == "compare":
                    if len(pdf_files) >= 2:
                        doc1_text = " ".join([doc.page_content for doc in all_docs if doc.metadata["source"] == pdf_files[0].name])
                        doc2_text = " ".join([doc.page_content for doc in all_docs if doc.metadata["source"] == pdf_files[1].name])
                        answer = handle_comparison(doc1_text, doc2_text, llm)
                    else:
                        answer = "⚠️ Please upload at least 2 PDFs to compare."

                elif task_type == "main_topic":
                    page_number = 3
                    page_texts = [doc.page_content for doc in all_docs if doc.metadata.get("page", 0) == page_number]
                    content = " ".join(page_texts)
                    answer = llm.invoke(f"What is the main topic of this content:\n\n{content}")

                else:
                    result = qa_chain.invoke({"question": query})
                    answer = result["answer"]
                    sources = set([doc.metadata.get("source") for doc in result["source_documents"]])
                    if sources:
                        answer += "\n\n📄 **Source(s)**: " + ", ".join(sources)

            st.session_state.chat.append(("bot", answer))

        # Display chat with expandable markdown and proper rendering
        for role, msg in st.session_state.chat:
            if role == "user":
                st.markdown(
                    f"""
                    <div style='text-align: right; margin: 8px 0;'>
                        <div style='display: inline-block; background-color: #f0f2f6; color: black;
                                    padding: 10px 15px; border-radius: 20px; max-width: 75%;'>
                            🧑‍🎓 {msg}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                with st.expander("📄 Click to expand response"):
                    st.markdown(msg.content if hasattr(msg, "content") else msg, unsafe_allow_html=True)

# 🔁 Reset chat button
st.markdown("---")
if st.button("🔁 Reset Chat"):
    st.session_state.chat = []
    if "memory" in st.session_state:
        st.session_state.memory.clear()
    st.success("✅ Chat history cleared!")
    time.sleep(1)
    st.rerun()
