%%writefile app.py
import streamlit as st
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

os.environ["GOOGLE_API_KEY"] = "AIzaSyBulf7B7HAfxQPPcCK6Q1ZoB4nc2kAGfJs"

@st.cache_resource
def setup_qa_chain():
    print("Setting up QA chain...")
    docs = []
    data_path = '/content/hackathon_data'
    if not os.path.exists(data_path):
        st.error(f"Data folder not found at {data_path}. Please make sure you uploaded and unzipped 'hackathon_data.zip' at the start of your session.")
        return None

    for file_name in os.listdir(data_path):
        file_path = os.path.join(data_path, file_name)
        if file_name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif file_name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)

    template = """You are the Kipi New Joiner Success Companion. Your goal is to provide helpful, concise, and accurate answers to new employees based ONLY on the provided context. If the answer is not in the context, state that you don't have enough information.
    Context: {context}
    Question: {question}
    Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print("QA chain setup complete.")
    return qa_chain

qa_chain = setup_qa_chain()

st.title("Kipi New Joiner Success Companion")
st.write("Hello! I'm here to help you navigate your onboarding journey at Kipi. Ask me anything!")

if "onboarding_stage" not in st.session_state:
    st.session_state.onboarding_stage = "Day 1-2" 

if "user_role" not in st.session_state:
    st.session_state.user_role = "General" 

st.sidebar.header("Your Onboarding Profile (Simulated)")
st.session_state.user_role = st.sidebar.selectbox(
    "Select your role:",
    ("General", "Delivery Analyst", "HR Generalist", "IT Support"),
    index=("General", "Delivery Analyst", "HR Generalist", "IT Support").index(st.session_state.user_role)
)
st.session_state.onboarding_stage = st.sidebar.selectbox(
    "Simulate your current onboarding stage:",
    ("Day 1-2", "Day 3-5", "Week 2-4", "Month 2-3"),
    index=("Day 1-2", "Day 3-5", "Week 2-4", "Month 2-3").index(st.session_state.onboarding_stage)
)
st.sidebar.markdown(f"**Current Role:** {st.session_state.user_role}")
st.sidebar.markdown(f"**Current Stage:** {st.session_state.onboarding_stage}")

st.sidebar.subheader("Your Onboarding Progress (Simulated)")
if st.session_state.onboarding_stage == "Day 1-2":
    st.sidebar.progress(25)
    st.sidebar.write("25% Complete - Focus on HR/IT Setup!")
elif st.session_state.onboarding_stage == "Day 3-5":
    st.sidebar.progress(50)
    st.sidebar.write("50% Complete - Explore Systems & Training!")
elif st.session_state.onboarding_stage == "Week 2-4":
    st.sidebar.progress(75)
    st.sidebar.write("75% Complete - Dive into Role-Specific Training!")
else: 
    st.sidebar.progress(90)
    st.sidebar.write("90% Complete - Almost there! Focus on Growth!")


st.sidebar.subheader("Quick Access (Synthetic Documents)")
st.sidebar.markdown("- **Laptop Welcome Kit Guide:** (Refer to this for IT setup)")
st.sidebar.markdown("- **New Hire Document Checklist:** (Important forms)")
st.sidebar.markdown("- **Expense Reimbursement Policy:** (For your claims)")
st.sidebar.markdown("- **New Joiner Onboarding Roadmap:** (Your journey guide)")
st.sidebar.markdown("- **IT Support FAQs:** (Common tech issues)")
st.sidebar.markdown("- **Kipi Leave Policy 2025:** (All about your time off)")
st.sidebar.markdown("- **Internal Comms Tools Guide:** (Slack, Teams, Email)")
st.sidebar.markdown("- **Kipi Resume Template Guidelines:** (For internal opportunities)")
st.sidebar.markdown("- **Employee Directory Access:** (Find your colleagues)")
st.sidebar.markdown("- **Learning Portal Access:** (Your growth journey)")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if qa_chain and (prompt_input := st.chat_input("Ask about onboarding, policies, or anything related to new joiners...")):
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    with st.chat_message("assistant"):
      st.write("---") 
st.markdown("**Was this helpful?**")
col1, col2 = st.columns([0.1, 0.9]) 
with col1:
    if st.button("👍", key=f"helpful_{len(st.session_state.messages)}"):
        st.toast("Thank you for your feedback!")
        
with col2:
    if st.button("👎", key=f"not_helpful_{len(st.session_state.messages)}"):
        feedback_text = st.text_input("What was confusing or missing?", key=f"feedback_input_{len(st.session_state.messages)}")
        if feedback_text:
            st.toast("Thanks for the feedback! We'll use this to improve our knowledge base.")
        with st.spinner("Finding your answer..."):
            try:
                
                full_prompt_for_ai = f"User Role: {st.session_state.user_role}. Onboarding Stage: {st.session_state.onboarding_stage}. Question: {prompt_input}"
                response = qa_chain.invoke({"query": full_prompt_for_ai})
                st.markdown(response["result"])
            except Exception as e:
                st.error(f"An error occurred: {e}. Please try again or check API key/data.")
                response = {"result": "Sorry, I couldn't find an answer based on the provided information."}
        st.session_state.messages.append({"role": "assistant", "content": response["result"]})
elif not qa_chain:
    st.warning("Companion not ready. Please ensure data is uploaded/unzipped and API key is set.")