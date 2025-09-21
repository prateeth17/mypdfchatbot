import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Load Google API key from Streamlit secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    logger.info("API key loaded from Streamlit secrets")
else:
    st.error("🚨 GOOGLE_API_KEY not found in secrets. Please add it to your Streamlit Cloud secrets.")
    st.write("Go to your Streamlit Cloud app settings → Secrets and add:")
    st.code('GOOGLE_API_KEY = "your-api-key-here"')
    st.stop()

# Page configuration
st.set_page_config(page_title="Chat with PDF", page_icon="📚")
st.title("Chat with your PDF 📚")

# Initialize session state variables
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processComplete" not in st.session_state:
    st.session_state.processComplete = None

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDF files"""
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Could not extract text from page {page_num}: {e}")
                    continue
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
        raise e
    
    return text

def get_text_chunks(text):
    """Split text into smaller chunks for processing"""
    # Even smaller chunks to prevent timeouts
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=300,   # Reduced from 500
        chunk_overlap=50, # Reduced overlap
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    # Limit number of chunks to prevent timeout
    max_chunks = 100
    if len(chunks) > max_chunks:
        st.warning(f"⚠️ Document has {len(chunks)} chunks. Using first {max_chunks} to prevent timeout.")
        chunks = chunks[:max_chunks]
    
    return chunks

def get_conversation_chain(vectorstore):
    """Create the conversation chain with the vector store"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-pro", 
            temperature=0.7,
            timeout=30  # Add timeout
        )
        
        template = """You are a helpful AI assistant that helps users understand their PDF documents.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer based on the provided context, just say that you don't know, don't try to make up an answer.
        Keep your answers concise and relevant to the question.
        
        Context: {context}
        
        Question: {question}
        Helpful Answer:"""

        prompt = PromptTemplate(
            input_variables=['context', 'question'], 
            template=template
        )
        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True,
            max_token_limit=2000  # Limit memory size
        )
        
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Limit retrieval results
            memory=memory,
            combine_docs_chain_kwargs={'prompt': prompt}
        )
        return conversation_chain
    except Exception as e:
        logger.error(f"Error creating conversation chain: {e}")
        raise e

def create_embeddings_with_retry(text_chunks, max_retries=5, initial_delay=2):
    """Create embeddings with exponential backoff retry logic"""
    
    # Process chunks in smaller batches
    batch_size = 10
    all_embeddings = []
    all_texts = []
    
    for batch_start in range(0, len(text_chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(text_chunks))
        batch_chunks = text_chunks[batch_start:batch_end]
        
        delay = initial_delay
        
        for attempt in range(1, max_retries + 1):
            try:
                st.info(f"Processing batch {batch_start//batch_size + 1}/{(len(text_chunks)-1)//batch_size + 1} (attempt {attempt})")
                
                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    task_type="retrieval_document"
                )
                
                # Create embeddings for this batch
                if attempt == 1:
                    # First attempt - create new vectorstore or add to existing
                    if not all_embeddings:
                        vectorstore = FAISS.from_texts(
                            texts=batch_chunks, 
                            embedding=embeddings
                        )
                    else:
                        # Add to existing vectorstore
                        batch_vectorstore = FAISS.from_texts(
                            texts=batch_chunks, 
                            embedding=embeddings
                        )
                        vectorstore.merge_from(batch_vectorstore)
                else:
                    # Retry - just this batch
                    if not all_embeddings:
                        vectorstore = FAISS.from_texts(
                            texts=batch_chunks, 
                            embedding=embeddings
                        )
                    else:
                        batch_vectorstore = FAISS.from_texts(
                            texts=batch_chunks, 
                            embedding=embeddings
                        )
                        vectorstore.merge_from(batch_vectorstore)
                
                all_texts.extend(batch_chunks)
                st.success(f"✅ Batch {batch_start//batch_size + 1} processed successfully")
                break  # Success - move to next batch
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Batch {batch_start//batch_size + 1} attempt {attempt} failed: {error_msg}")
                
                if "quota" in error_msg.lower() or "rate" in error_msg.lower():
                    st.warning(f"⚠️ Rate limit hit. Waiting {delay} seconds...")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                elif "timeout" in error_msg.lower() or "deadline" in error_msg.lower():
                    st.warning(f"⚠️ Timeout occurred. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= 1.5
                else:
                    st.warning(f"⚠️ Embedding error: {error_msg}")
                    time.sleep(delay)
                
                if attempt == max_retries:
                    st.error(f"❌ Failed to process batch {batch_start//batch_size + 1} after {max_retries} attempts.")
                    st.error("Try with a smaller PDF or fewer files.")
                    return None
    
    return vectorstore if all_texts else None

def process_docs(pdf_docs):
    """Process uploaded PDF documents"""
    try:
        # Validate file size
        total_size = sum(pdf.size for pdf in pdf_docs if hasattr(pdf, 'size'))
        max_size = 10 * 1024 * 1024  # 10MB limit
        
        if total_size > max_size:
            st.error(f"❌ Total file size ({total_size/1024/1024:.1f}MB) exceeds limit (10MB). Please upload smaller files.")
            return False
        
        # Get PDF text
        with st.spinner("📄 Extracting text from PDFs..."):
            raw_text = get_pdf_text(pdf_docs)
        
        if not raw_text.strip():
            st.error("❌ No text found in the uploaded PDFs. Please ensure your PDFs contain extractable text.")
            return False
        
        st.info(f"📊 Extracted {len(raw_text)} characters from your PDFs")
        
        # Get text chunks
        with st.spinner("✂️ Splitting text into chunks..."):
            text_chunks = get_text_chunks(raw_text)
        
        st.info(f"📝 Created {len(text_chunks)} text chunks")
        
        # Create embeddings with retry
        with st.spinner("🧠 Creating embeddings (this may take a while)..."):
            vectorstore = create_embeddings_with_retry(text_chunks)
        
        if not vectorstore:
            return False
        
        # Create conversation chain
        with st.spinner("🔗 Setting up conversation chain..."):
            st.session_state.conversation = get_conversation_chain(vectorstore)
        
        st.session_state.processComplete = True
        return True
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        st.error(f"❌ An error occurred during processing: {str(e)}")
        return False

# Sidebar for PDF upload
with st.sidebar:
    st.subheader("📁 Your Documents")
    
    pdf_docs = st.file_uploader(
        "Upload your PDFs here (max 10MB total)",
        type="pdf",
        accept_multiple_files=True,
        help="Select one or more PDF files to chat with"
    )
    
    if pdf_docs:
        st.write(f"📄 {len(pdf_docs)} file(s) selected")
        total_size = sum(getattr(pdf, 'size', 0) for pdf in pdf_docs)
        st.write(f"📊 Total size: {total_size/1024/1024:.1f}MB")
    
    process_button = st.button("🚀 Process Documents", disabled=not pdf_docs)
    
    if process_button and pdf_docs:
        with st.spinner("Processing your PDFs..."):
            success = process_docs(pdf_docs)
            if success:
                st.success("✅ Processing complete! You can now chat with your documents.")
    
    # Reset button
    if st.button("🔄 Reset Chat"):
        st.session_state.conversation = None
        st.session_state.chat_history = []
        st.session_state.processComplete = None
        st.rerun()

# Main chat interface
if st.session_state.processComplete:
    st.success("✅ Documents processed! Ask questions below:")
    
    user_question = st.chat_input("💬 Ask a question about your documents...")
    
    if user_question:
        try:
            # Add user message to chat history immediately
            st.session_state.chat_history.append(("user", user_question))
            
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.conversation({
                    "question": user_question
                })
                
                # Add bot response to chat history
                st.session_state.chat_history.append(("assistant", response["answer"]))
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            st.error(f"❌ An error occurred during chat: {str(e)}")
            st.session_state.chat_history.append(("assistant", "Sorry, I encountered an error processing your question. Please try again."))

    # Display chat history (most recent first)
    for i in range(len(st.session_state.chat_history) - 1, -1, -1):
        role, message = st.session_state.chat_history[i]
        with st.chat_message(role):
            st.write(message)

# Display initial instructions
else:
    st.markdown("""
    ### 🚀 Getting Started
    
    1. **Upload PDFs**: Use the sidebar to upload your PDF documents
    2. **Process**: Click the "Process Documents" button
    3. **Chat**: Ask questions about your documents!
    
    ### ⚠️ Tips for best results:
    - Keep total file size under 10MB
    - Use PDFs with extractable text (not scanned images)
    - Be patient during processing - embedding creation can take time
    - If you get timeout errors, try smaller files or fewer documents
    """)

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: For large documents, consider splitting them into smaller files to avoid timeout errors.")
