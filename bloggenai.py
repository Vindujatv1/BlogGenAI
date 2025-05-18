import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import base64
import streamlit.components.v1 as components
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.utilities import SerpAPIWrapper 
import hashlib
import traceback
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# IMPORTANT: set_page_config must be the first Streamlit command
st.set_page_config(
    page_title="BlogGen AI",
    page_icon="✍️",
    layout="wide"
)

# Load environment variables - do this before other Streamlit commands
load_dotenv()
serpapi_api_key = os.getenv("SERPAPI_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

# Set Google API key for Gemini
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
else:
    st.error("⚠️ GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Check if SerpAPI key is available
if not serpapi_api_key:
    st.error("⚠️ SERPAPI_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .title-container {
        background-color: #f5f5f5;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .blog-output {
        background-color: #f9f9f9;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        margin-top: 1.5rem;
    }
    .sidebar .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .source-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        background-color: #e0f7fa;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .trusted-source {
        background-color: #e8f5e9;
    }
</style>
""", unsafe_allow_html=True)

# App title and description
st.markdown("""
<div class="title-container">
    <h1>BlogGen AI</h1>
    <p>Generate high-quality, SEO-optimized blog posts using Gemini AI with enhanced web research</p>
</div>
""", unsafe_allow_html=True)

# Trusted domains and spam patterns
TRUSTED_DOMAINS = ['.gov', '.edu', '.org', 'wikipedia.org', 'nytimes.com', 'bbc.com', 'reuters.com']
SPAM_KEYWORDS = ['buy now', 'discount', 'limited offer', 'click here', 'cheap', 'viagra']

# Custom SerpAPI wrapper with better result handling
class EnhancedSerpAPIWrapper:
    def __init__(self, serpapi_api_key: str):
        self.serpapi = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    
    def search(self, query: str, num_results: int = 10) -> List[Dict]:
        """Perform search and return structured results"""
        try:
            # SerpAPI doesn't directly support num_results, but we can request more and filter
            results = self.serpapi._process_response(self.serpapi.run(query))
            
            # Extract organic results
            organic_results = results.get('organic_results', [])
            
            # Convert to standard format
            formatted_results = []
            for res in organic_results:
                formatted = {
                    'title': res.get('title', ''),
                    'snippet': res.get('snippet', ''),
                    'link': res.get('link', ''),
                    'position': res.get('position', 0)
                }
                formatted_results.append(formatted)
            
            return formatted_results[:num_results*2]  # Return extra for filtering
        
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return []

# Initialize components
@st.cache_resource
def load_models():
    # Initialize Enhanced SerpAPI wrapper
    serpapi = EnhancedSerpAPIWrapper(serpapi_api_key=serpapi_api_key)
    
    # Initialize Gemini model
    gemini_model = ChatGoogleGenerativeAI(temperature=0.3, model="gemini-2.0-flash")
    
    # Initialize Sentence Transformer for semantic similarity
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return serpapi, gemini_model, embedding_model

# Function to filter and score search results
def filter_and_score_results(results: List[Dict], query: str, embedding_model) -> List[Dict]:
    filtered_results = []
    
    for result in results:
        # Skip if no snippet or title
        if not result.get('snippet') or not result.get('title'):
            continue
            
        # Check domain trustworthiness
        domain = result.get('link', '').split('/')[2] if result.get('link') else ''
        is_trusted = any(td in domain for td in TRUSTED_DOMAINS)
        
        # Check for spam indicators
        text = f"{result['title']} {result['snippet']}".lower()
        is_spam = any(spam_keyword in text for spam_keyword in SPAM_KEYWORDS)
        
        if is_spam:
            continue
            
        # Add trust score
        result['trust_score'] = 1.0 if is_trusted else 0.5
        filtered_results.append(result)
    
    if not filtered_results:
        return filtered_results
    
    # Calculate semantic similarity scores
    query_embedding = embedding_model.encode([query])
    result_texts = [f"{res['title']} {res['snippet']}" for res in filtered_results]
    result_embeddings = embedding_model.encode(result_texts)
    
    similarity_scores = cosine_similarity(query_embedding, result_embeddings)[0]
    
    # Combine scores (50% semantic similarity, 30% trust score, 20% position)
    for i, res in enumerate(filtered_results):
        semantic_score = similarity_scores[i]
        position_score = 1 - (i / len(filtered_results))  # Higher for earlier results
        combined_score = (0.5 * semantic_score) + (0.3 * res['trust_score']) + (0.2 * position_score)
        res['combined_score'] = combined_score
    
    # Sort by combined score
    filtered_results.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return filtered_results

# Function to process text into chunks
def process_text_into_chunks(raw_text: str, max_chunk_size: int = 300) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chunk_size,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = splitter.create_documents([raw_text])
    return [doc.page_content for doc in docs]

# Estimate token count (approximate)
def count_tokens(text: str) -> int:
    return len(text.split())

# Function to generate blog with enhanced web search
def generate_blog_with_search(topic: str, no_words: int, blog_style: str, tone: str, 
                            age_group: str, keywords: str, blog_format: str, 
                            num_results: int, min_trust_score: float = 0.6) -> str:
    try:
        # Load models
        serpapi, gemini_model, embedding_model = load_models()

        with st.spinner("Searching the web for relevant information..."):
            # Get raw search results
            raw_results = serpapi.search(topic, num_results=num_results*2)  # Get extra to account for filtering
            
            # Filter and re-rank results
            filtered_results = filter_and_score_results(raw_results, topic, embedding_model)
            
            # Apply trust score threshold
            trusted_results = [res for res in filtered_results if res['combined_score'] >= min_trust_score][:num_results]
            
            
            # Display sources used
            with st.expander("ℹ️ Sources used in this blog"):
                for res in trusted_results:
                    domain = res.get('link', '').split('/')[2] if res.get('link') else 'unknown'
                    trust_class = "trusted-source" if res['trust_score'] >= 0.8 else ""
                    st.markdown(
                        f"""
                        <div class="source-badge {trust_class}" title="Trust score: {res['trust_score']:.2f}">
                            {domain}
                        </div>
                        <strong>{res['title']}</strong><br>
                        {res.get('snippet', 'No snippet available')}<br>
                        <small><a href="{res.get('link', '#')}" target="_blank">Source</a></small>
                        <hr>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Combine content from trusted results
            raw_text = "\n\n".join([
                f"Source: {res.get('link', 'Unknown')}\n"
                f"Title: {res['title']}\n"
                f"Content: {res.get('snippet', 'No content')}"
                for res in trusted_results
            ])

        with st.spinner("Processing information..."):
            # Process text into chunks
            chunks = process_text_into_chunks(raw_text)
            context = "\n\n".join(chunks)

        keywords_section = f"Include the following keywords: {keywords}." if keywords else ""

        prompt = f""" 
        You are an expert blogger writing for {blog_style.lower()} audience aged {age_group.lower()}. 
        Using the information below extracted from reliable web sources:

        {context}

        Write a {tone.lower()} blog about \"{topic}\" that is approximately {no_words} words long. 
        Start the blog with a title (H1), followed by a short SEO meta-description (less than 160 characters). 
        Include a list of 5-8 keywords at the end under a heading 'SEO Keywords'.
        The blog should be formatted using {blog_format.lower()}. {keywords_section}

        Structure the blog with:
        - An engaging introduction
        - 3-4 key points or sections with explanations
        - A clear conclusion summarizing the topic
        - Citations to sources where appropriate

        Important guidelines:
        1. Verify facts from multiple sources before including them
        2. Clearly distinguish between facts and opinions
        3. Cite sources for statistics and specific claims
        4. Maintain a neutral, professional tone unless humor is specifically requested

        Write the blog now:
        """

        with st.spinner("Generating your blog with Gemini AI..."):
            response = gemini_model.invoke(prompt)

        return response.content
    except Exception as e:
        st.error(f"❌ Error generating blog: {str(e)}")
        st.text("Traceback:")
        st.code(traceback.format_exc())
        return f"An error occurred while generating your blog. Please try again with different parameters.\nError details: {str(e)}"

# Create two columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Blog Parameters")

    # Input fields
    topic = st.text_input("Blog Topic", placeholder="e.g., The Future of Renewable Energy")

    no_words = st.number_input("Approximate Word Count", 
                             min_value=300, 
                             max_value=2000, 
                             value=800, 
                             step=100)

    blog_style = st.selectbox("Blog Style", 
                             ["Professional", "Casual", "Educational", "Technical", "Conversational"])

    tone = st.selectbox("Tone", 
                       ["Informative", "Persuasive", "Inspirational", "Critical", "Humorous"])

    age_group = st.selectbox("Target Age Group", 
                            ["18-24", "25-34", "35-44", "45-54", "55+", "General Audience"])

    keywords = st.text_input("SEO Keywords (comma separated)", 
                            placeholder="e.g., sustainable energy, green technology")

    blog_format = st.selectbox("Format", 
                              ["Markdown", "HTML", "Plain Text"])

    num_results = st.slider("Number of Web Results to Use", min_value=1, max_value=10, value=5)
    
    min_trust_score = st.slider("Minimum Source Trust Score", 
                               min_value=0.1, 
                               max_value=1.0, 
                               value=0.6, 
                               step=0.1,
                               help="Higher values prioritize .gov, .edu, and reputable news sources")

    generate_button = st.button("Generate Blog", type="primary", use_container_width=True)

# Output column for displaying the blog
with col2:
    st.subheader("Generated Blog")
    blog_container = st.empty()

    if generate_button and topic:
        try:
            blog_content = generate_blog_with_search(
                topic=topic,
                no_words=no_words,
                blog_style=blog_style,
                tone=tone,
                age_group=age_group,
                keywords=keywords,
                blog_format=blog_format,
                num_results=num_results,
                min_trust_score=min_trust_score
            )

            token_count = count_tokens(blog_content)
            st.success(f"✅ Approximate token usage: {token_count} tokens")

            with blog_container.expander("📄 Click to view the generated blog", expanded=True):
                st.markdown(f"""
                <div class="blog-output">
                    {blog_content}
                </div>
                """, unsafe_allow_html=True)

            blog_bytes = blog_content.encode()
            b64 = base64.b64encode(blog_bytes).decode()
            filename = f"{topic.replace(' ', '_')}_blog.txt"
            href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Blog as Text File</a>'
            st.markdown(href, unsafe_allow_html=True)

            st.code(blog_content, language="markdown")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        blog_container.info("Your generated blog will appear here. Fill in the parameters and click 'Generate Blog'.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit, Langchain, Google Gemini AI, and enhanced web research")