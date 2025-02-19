import streamlit as st
import requests
from openai import OpenAI
import google.generativeai as genai
from firecrawl.firecrawl import FirecrawlApp
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
SEARXNG_INSTANCE = os.getenv('SEARXNG_INSTANCE')
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
FIRECRAWL_ENDPOINT = os.getenv('FIRECRAWL_ENDPOINT')
SEARCH_WIDTH = 1
SEARCH_DEPTH = 1
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize APIs
firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY, api_url=FIRECRAWL_ENDPOINT)

# Session state initialization
if 'learnings' not in st.session_state:
    st.session_state.learnings = []
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'depth' not in st.session_state:
    st.session_state.depth = 0

def setup_llm_providers():
    """Initialize LLM providers based on sidebar selection"""
    if st.session_state.llm_provider == "OpenAI":
        return OpenAI(base_url=st.session_state.openai_base, api_key=st.session_state.openai_key)
    elif st.session_state.llm_provider == "Google":
        genai.configure(api_key=st.session_state.google_key)
        return genai
    return None

def generate_llm_response(prompt, system_prompt=None, task_type="reasoning"):
    """Handle different LLM providers and model types"""
    model_name = st.session_state.reasoning_model if task_type == "reasoning" else st.session_state.summarization_model
    
    if st.session_state.llm_provider == "OpenAI":
        client = setup_llm_providers()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7 if task_type == "reasoning" else 0.3
        )
        return response.choices[0].message.content.strip()
    
    elif st.session_state.llm_provider == "Google":
        client = setup_llm_providers()
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text

def clarify_query(original_query):
    clarification_prompt = f"""
    The user submitted this research query: {original_query}
    What additional information would help better understand their needs?
    Ask up to 2 clarifying questions, be concise and direct.
    """
    print("Clarifying")
    return generate_llm_response(clarification_prompt, task_type="reasoning")

def generate_search_queries(research_context):
    prompt = f"""
    Based on the research context below, generate {SEARCH_WIDTH} diverse search queries 
    to gather comprehensive information. Focus on different aspects and perspectives.
    
    Research Context: {research_context}
    
    Return each query on a new line without numbering.
    """
    search_queries = generate_llm_response(prompt, task_type="reasoning")
    print(f"Search request: {search_queries}")
    return search_queries

def search_searxng(query):
    params = {
        "q": query,
        "format": "json",
        "safesearch": 1,
        "categories": "general",
        "language": "en",
        "page": 1
    }
    print(f"Searching for: {query}")
    try:
        response = requests.get(f"{SEARXNG_INSTANCE}/search", params=params)
        return response.json()["results"][:5]
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def scrape_page(url):
    print(f"Scraping page for: {url}")
    try:
        scraped = firecrawl.scrape_url(url)
        print(scraped)
        return {
            "content": scraped['markdown'] or scraped['text'],
            "url": url
        }
    except Exception as e:
        st.error(f"Firecrawl error: {e}")
        return None

def analyze_content(content, research_goal):
    prompt = f"""
    Analyze this content in relation to: {research_goal}
    Extract key insights, facts, statistics, and relevant information.
    Identify potential gaps needing further investigation.
    
    Content: {content[:12000]}
    
    Provide a concise summary of learnings.
    """
    return generate_llm_response(prompt, task_type="summarization")

def generate_report(learnings, sources, research_goal):
    prompt = f"""
    Compile these key learnings into a comprehensive research report about: {research_goal}
    Structure with clear sections and include source citations using [number] notation.
    
    Learnings: {"\n".join(learnings)}
    
    Include a '## Sources' section at the end listing all referenced URLs.
    """
    report = generate_llm_response(prompt, task_type="summarization")
    return f"{report}\n\n## Sources\n" + "\n".join(
        f"[{i+1}] {url}" for i, url in enumerate(sources)
    )

def sidebar_config():
    with st.sidebar:
        st.header("Configuration")
        llm_provider = st.selectbox(
            "LLM Provider",
            ["OpenAI", "Google"],
            key="llm_provider",
            index=1
        )
        
        st.subheader("Model Selection")
        st.selectbox(
            "Reasoning Model",
            ["gpt-4", "llama3-70b", "gemini-1.5-pro", "gemini-2.0-pro-exp-02-05", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash"],
            key="reasoning_model",
            index=5
        )
        st.selectbox(
            "Summarization Model",
            ["gpt-3.5-turbo", "llama3-8b", "gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05"],
            key="summarization_model",
            index=2
        )
        
        st.subheader("API Keys")
        if llm_provider == "OpenAI":
            st.text_input("OpenAI Base URL", key="openai_base", value="http://localhost:11434/v1")
            st.text_input("OpenAI API Key", key="openai_key", type="password")
        elif llm_provider == "Google":
            st.text_input("Google API Key", key="google_key", type="password", value=GEMINI_API_KEY)

def main_app():
    st.title("Deep Research Assistant")
    user_query = st.text_input("Enter your research question:", value="hello")

    if user_query:
        if "clarified" not in st.session_state:
            clarification = clarify_query(user_query)
            st.info(clarification)
            clarification_answer = st.text_input("Please provide additional details:", value="how to say it in 4 languages")
            if clarification_answer:
                user_query += f" {clarification_answer}"
                st.session_state.clarified = True
                st.rerun()
            return

        if st.session_state.depth < SEARCH_DEPTH:
            st.write(f"## Search Iteration {st.session_state.depth + 1}")
            search_queries = generate_search_queries(user_query)
            
            for query in search_queries.splitlines():
                st.write(f"**Searching:** `{query}`")
                results = search_searxng(query)
                
                for result in results:
                    with st.expander(f"{result['title']} - {result['url']}"):
                        scraped = scrape_page(result['url'])
                        if scraped and scraped['content']:
                            st.session_state.sources.append(scraped['url'])
                            learning = analyze_content(scraped['content'], user_query)
                            st.session_state.learnings.append(learning)
                            st.write(f"**Key Learning:** {learning}")
            
            st.session_state.depth += 1
            st.rerun()
        else:
            report = generate_report(
                st.session_state.learnings,
                list(set(st.session_state.sources)),
                user_query
            )
            st.write("## Final Research Report")
            st.markdown(report)
            
            # Reset session
            #for key in ['learnings', 'sources', 'depth', 'clarified']:
               # if key in st.session_state:
                    #del st.session_state[key]

if __name__ == "__main__":
    sidebar_config()
    main_app()