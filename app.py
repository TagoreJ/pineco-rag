import streamlit as st
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Universal RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dynamic Theme CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Base Styles */
* { 
    font-family: 'Inter', sans-serif; 
}
.stApp { 
    background-color: var(--secondary-background-color); 
    color: var(--text-color);
}

/* Sidebar Styling is naturally handled by Streamlit, but we can tweak borders */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.2);
}
[data-testid="stSidebar"] hr {
    margin: 1rem 0;
    border-color: rgba(128, 128, 128, 0.2);
}
[data-testid="stSidebar"] .stMarkdown h2, 
[data-testid="stSidebar"] .stMarkdown h3 {
    font-weight: 600;
}

/* Primary Connect Button */
.connect-btn > button {
    background: var(--primary-color) !important;
    color: white !important;
    border: none !important;
    width: 100% !important;
    padding: 0.6rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.05) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}
.connect-btn > button:hover {
    filter: brightness(1.1);
    transform: translateY(-1px);
}

/* Header Area */
.header-container {
    background: var(--background-color);
    padding: 2rem;
    border-radius: 12px;
    margin-bottom: 2rem;
    border: 1px solid rgba(128, 128, 128, 0.2);
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.header-icon {
    font-size: 3rem;
    background: rgba(128, 128, 128, 0.1);
    padding: 1rem;
    border-radius: 16px;
    color: var(--primary-color);
}
.header-text h1 { 
    font-size: 1.75rem; 
    font-weight: 700; 
    margin: 0; 
}
.header-text p { 
    opacity: 0.8; 
    margin: 0.5rem 0 0; 
    font-size: 1rem; 
}

/* Chat Messages */
.chat-container {
    max-width: 800px;
    margin: 0 auto;
}
.user-bubble {
    background: var(--primary-color);
    color: white;
    padding: 1rem 1.25rem;
    border-radius: 16px 16px 4px 16px;
    margin: 0.5rem 0 0.5rem auto;
    max-width: 80%;
    width: fit-content;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}
.bot-bubble {
    background: var(--background-color);
    color: var(--text-color);
    padding: 1.25rem 1.5rem;
    border-radius: 16px 16px 16px 4px;
    border: 1px solid rgba(128, 128, 128, 0.2);
    margin: 0.5rem 0;
    max-width: 85%;
    width: fit-content;
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}

/* Sources & Metadata */
.source-section {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(128, 128, 128, 0.2);
    font-size: 0.85rem;
    opacity: 0.8;
}
.source-badge {
    background: rgba(128, 128, 128, 0.1);
    border: 1px solid rgba(128, 128, 128, 0.2);
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    margin-right: 0.5rem;
    display: inline-block;
    font-family: monospace;
}

/* Status Indicators */
.status-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.75rem 1rem;
    border-radius: 8px;
    font-weight: 500;
    margin-bottom: 1rem;
}
.status-connected {
    background: rgba(34, 197, 94, 0.1);
    color: #16a34a; /* slightly tailored green */
    border: 1px solid rgba(34, 197, 94, 0.2);
}
.status-disconnected {
    background: rgba(239, 68, 68, 0.1);
    color: #dc2626; /* slightly tailored red */
    border: 1px solid rgba(239, 68, 68, 0.2);
}
.status-dot {
    width: 8px; 
    height: 8px;
    border-radius: 50%;
}
.status-dot.online { background: #22c55e; box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2); }
.status-dot.offline { background: #ef4444; box-shadow: 0 0 0 2px rgba(239, 68, 68, 0.2); }

/* Stats Cards */
.stats-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.stat-card {
    background: var(--background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 8px;
    padding: 0.75rem;
    text-align: center;
}
.stat-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--primary-color);
}
.stat-label {
    font-size: 0.75rem;
    opacity: 0.7;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 0.25rem;
}

/* Welcome Card */
.welcome-card {
    background: var(--background-color);
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.welcome-card h3 {
    margin-top: 0;
}
.step-list {
    margin-top: 1rem;
}
.step-item {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    margin-bottom: 1rem;
}
.step-number {
    background: rgba(128, 128, 128, 0.1);
    color: var(--primary-color);
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 0.85rem;
    flex-shrink: 0;
}
</style>
""", unsafe_allow_html=True)


# ── State Management ──────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "connected": False,
        "embeddings": None,
        "index": None,
        "llm": None,
        "tokens_used": 0,
        "msg_count": 0,
        "system_prompt": """You are a helpful, professional AI assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain the answer, politely state that you cannot answer based on the provided documents.
Be clear, concise, and format your response well using markdown.

History: {history}

Context:
{context}

Question: {question}

Answer:"""
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── Sidebar Configuration ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # Status display at the top of sidebar
    if st.session_state.connected:
        st.markdown("""
        <div class="status-indicator status-connected">
            <div class="status-dot online"></div>
            <span>System Connected</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-indicator status-disconnected">
            <div class="status-dot offline"></div>
            <span>System Disconnected</span>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")

    with st.expander("Provider Credentials", expanded=not st.session_state.connected):
        groq_key = st.text_input("Groq API Key", type="password", help="Required for LLM generation")
        pinecone_key = st.text_input("Pinecone API Key", type="password", help="Required for Vector DB access")
        
    with st.expander("Database Settings", expanded=not st.session_state.connected):
        index_name = st.text_input("Pinecone Index Name")
        top_k = st.number_input("Documents to Retrieve (Top K)", min_value=1, max_value=20, value=5)
        min_score = st.slider("Minimum Relevance Score", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
    with st.expander("Advanced Output Settings"):
        st.session_state.system_prompt = st.text_area(
            "System Prompt Template", 
            value=st.session_state.system_prompt,
            height=200,
            help="Customize how the AI responds. Must include {history}, {context}, and {question} placeholders."
        )

    st.markdown('<div class="connect-btn">', unsafe_allow_html=True)
    if st.button("Initialize Pipeline", use_container_width=True):
        if not all([groq_key, pinecone_key, index_name]):
            st.error("Please provide all required credentials and index name.")
        else:
            with st.spinner("Initializing components..."):
                try:
                    # 1. Init Embeddings
                    st.session_state.embeddings = FastEmbedEmbeddings(
                        model_name="BAAI/bge-small-en-v1.5"
                    )
                    
                    # 2. Init Vector DB
                    pc = Pinecone(api_key=pinecone_key)
                    st.session_state.index = pc.Index(index_name)
                    
                    # 3. Init LLM
                    st.session_state.llm = ChatGroq(
                        model_name="llama-3.1-8b-instant",
                        temperature=0.1,  # Lower temperature for more factual RAG
                        max_tokens=1024,
                        api_key=groq_key
                    )
                    
                    # 4. Verify connection (attempt fetch stats)
                    _ = st.session_state.index.describe_index_stats()
                    
                    st.session_state.connected = True
                    st.success("Successfully initialized!")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.session_state.connected = False
                    st.error(f"Initialization failed: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.connected:
        st.markdown("---")
        st.markdown("### Session Statistics")
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{st.session_state.msg_count}</div>
                <div class="stat-label">Interactions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{st.session_state.tokens_used:,}</div>
                <div class="stat-label">Est. Tokens</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Clear Conversation", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.session_state.msg_count = 0
            st.session_state.tokens_used = 0
            st.rerun()

# ── Core Functions ────────────────────────────────────────────────────────────
def build_prompt():
    return ChatPromptTemplate.from_template(st.session_state.system_prompt)

def process_query(query: str):
    """Handles embedding, retrieval, and generation."""
    try:
        # 1. Embed query
        vector = st.session_state.embeddings.embed_query(query)
        
        # 2. Retrieve
        results = st.session_state.index.query(
            vector=vector, 
            top_k=top_k, 
            include_metadata=True
        )
        
        if not results.get("matches"):
            return "Cannot retrieve documents. The index might be empty.", []
            
        # 3. Process matches
        contexts = []
        sources = []
        
        for match in results["matches"]:
            score = match.get("score", 0)
            if score < min_score:
                continue
                
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            
            # Construct a citation string dynamically based on available metadata
            src_parts = []
            if "source" in metadata or "source_file" in metadata:
                src_parts.append(metadata.get("source") or metadata.get("source_file"))
            if "page" in metadata:
                src_parts.append(f"Page {metadata['page']}")
            if "chunk" in metadata:
                src_parts.append(f"Chunk {metadata['chunk']}")
                
            citation = " | ".join(src_parts) if src_parts else "Unknown Source"
            
            contexts.append(f"[Source: {citation}, Relevance: {score:.2f}]\n{text}")
            sources.append(citation)
            
        if not contexts:
            return "No information found in the database that meets the minimum relevance threshold.", []
            
        combined_context = "\n\n---\n\n".join(contexts)
        unique_sources = list(set(sources))
        
        # 4. Prepare History
        history = "\n".join([
            f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
            for m in st.session_state.messages[-4:] # Keep last 2 turns to save context window
        ])
        
        # 5. Generate
        prompt = build_prompt()
        messages = prompt.format_messages(
            history=history or "No previous history.",
            context=combined_context,
            question=query
        )
        
        response = st.session_state.llm.invoke(messages)
        
        # 6. Track metrics
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            st.session_state.tokens_used += response.usage_metadata.get("total_tokens", 0)
            
        return response.content, unique_sources
        
    except Exception as e:
        return f"An error occurred during processing: {str(e)}", []


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='header-container'>
    <div class='header-icon'>📚</div>
    <div class='header-text'>
        <h1>Universal Knowledge Assistant</h1>
        <p>Connect your Pinecone Vector Database and chat with your documents Instantly.</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.connected:
    st.markdown("""
    <div class="welcome-card">
        <h3>👋 Welcome! Let's get started.</h3>
        <p style="color: #64748b; margin-bottom: 2rem;">Configure your environment to start querying your documents.</p>
        
        <div class="step-list">
            <div class="step-item">
                <div class="step-number">1</div>
                <div>
                    <strong>API Credentials</strong><br>
                    <span style="color: #64748b; font-size: 0.9rem;">Enter your Groq API key for the LLM and your Pinecone API key for vector storage in the sidebar.</span>
                </div>
            </div>
            <div class="step-item">
                <div class="step-number">2</div>
                <div>
                    <strong>Database Target</strong><br>
                    <span style="color: #64748b; font-size: 0.9rem;">Specify the exact name of your Pinecone index where your vectors are stored.</span>
                </div>
            </div>
            <div class="step-item">
                <div class="step-number">3</div>
                <div>
                    <strong>Initialize</strong><br>
                    <span style="color: #64748b; font-size: 0.9rem;">Click the Initialize Pipeline button to establish connections and load the embedding model.</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Chat Interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Render history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        sources_html = ""
        if msg.get("sources"):
            badges = "".join([f"<span class='source-badge'>{src}</span>" for src in msg['sources']])
            sources_html = f"<div class='source-section'><strong>Sources:</strong><br>{badges}</div>"
            
        st.markdown(f"<div class='bot-bubble'>{msg['content']}{sources_html}</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.msg_count += 1
    st.rerun()

# Handle pending response
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    
    with st.spinner("Analyzing documents to formulate answer..."):
        answer, sources = process_query(user_query)
        
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer, 
        "sources": sources
    })
    st.rerun()
