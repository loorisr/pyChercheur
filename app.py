import streamlit as st
import requests
import litellm
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

# Initialize APIs
firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY, api_url=FIRECRAWL_ENDPOINT)

# Session state initialization
if 'learnings' not in st.session_state:
    st.session_state.learnings = []
if 'sources' not in st.session_state:
    st.session_state.sources = []
if 'depth' not in st.session_state:
    st.session_state.depth = 0
if 'reasoning_model' not in st.session_state:
    st.session_state.reasoning_model = "gemini-2.0-flash"  # Default reasoning model
if 'summarization_model' not in st.session_state:
    st.session_state.summarization_model = "gemini-2.0-flash"  # Default summarization model
if 'llm_provider' not in st.session_state:
    st.session_state.llm_provider = "gemini"  # Default LLM provider

def generate_llm_response(prompt, system_prompt=None, task_type="reasoning"):
    """Generate response using LiteLLM."""
    model_name = st.session_state.reasoning_model if task_type == "reasoning" else st.session_state.summarization_model
    model_name = st.session_state.llm_provider + "/" + model_name
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    #print(f"Call {model_name} with {messages}")
    
    try:
        response = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=0.7 if task_type == "reasoning" else 0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return ""

def clarify_query(original_query):
    clarification_prompt = f"""
    The user submitted this research query: {original_query}
    What additional information would help better understand their needs?
    Ask up to 2 clarifying questions, be concise and direct.
    """
    return generate_llm_response(clarification_prompt, task_type="reasoning")

def generate_search_queries(research_context):
    prompt = f"""
    Based on the research context below, generate {SEARCH_WIDTH} diverse search queries 
    to answer to it.
    
    Research Context: {research_context}
    
    Return each query on a new line without numbering.
    """
    search_queries = generate_llm_response(prompt, task_type="reasoning")
    return search_queries.splitlines()

def check_page_to_scrape(url, content, research_goal):  ################
    prompt = f"""
    Analyze this url {url} in relation to: {research_goal}.
    The summury of this url is {content}.
    Tell by true or false if it would be interesting to get the complete page or if we can skip it.
    
    Respond only with true or false.
    """
    return generate_llm_response(prompt, task_type="summarization").splitlines()


def list_pages_to_scrape(search_results, research_goal):  ################
    prompt = f"""
    Analyze theses searches results (url and content) in relation to: {research_goal}.
    Select only the 3 urls that are the most relevant to answer to the research.

    Search results: {search_results}
    
    Return each url that you have selected on a new line without numbering.
    """
    return generate_llm_response(prompt, task_type="summarization")

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
        urls = [item["url"] for item in response.json()["results"][:5]]
        #print(f"results: {urls}")
        return response.json()["results"]#[:5]
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def scrape_page(url):
    print(f"Scraping page for: {url}")
    try:
        scraped = firecrawl.scrape_url(url)
        return {
            "content": scraped['markdown'],
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

def settings_page():
    st.title("Settings")
    st.header("LLM Configuration")
    
    llm_provider = st.selectbox(
        "LLM Provider",
        ["openai", "gemini", "ollama", "mistral"],
        key="llm_provider",
        index=0
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
        ["gemma2:2b", "llama3-8b", "gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05"],
        key="summarization_model",
        index=2
    )
    
    st.subheader("API Keys")
    api_key = st.text_input(f"{llm_provider.capitalize()} API Key", 
                          type="password",
                          key=f"{llm_provider}_key")
    
    if api_key:
        os.environ[f"{llm_provider.upper()}_API_KEY"] = api_key


def get_content_by_url(data_list, target_url):
    # Iterate through the list to find the matching URL
    for item in data_list:
        if item["url"] == target_url:
            return item
    # Return None if no match is found
    return None

def main_app():
    st.title("Deep Research Assistant")
    user_query = st.text_input("Enter your research question:", value="hello")

    if user_query:
        if "clarified" not in st.session_state:
            clarification = clarify_query(user_query)
            st.info(clarification)
            clarification_answer = st.text_input("Please provide additional details:", value="how to say in 5 languages")
            if clarification_answer:
                user_query += f" {clarification_answer}"
                st.session_state.clarified = True
                st.rerun()
            return

        if st.session_state.depth < SEARCH_DEPTH:
            st.write(f"## Search Iteration {st.session_state.depth + 1}")
            search_queries = generate_search_queries(user_query)
            search_queries = search_queries[:SEARCH_WIDTH]
            st.info(search_queries)
            print(f"Generated search queries {search_queries}")
            
            for query in search_queries:
                st.write(f"**Searching:** `{query}`")
                results = search_searxng(query)
                for result in results:
                    st.info(f"**{result['title']}**  \n{result['url']}")
                
                urls_to_scrape = list_pages_to_scrape(results, user_query)
                print(urls_to_scrape)
                #print(results)

                # Display search results
                for url in urls_to_scrape:
                    print(url)
                    result = get_content_by_url(results, url)
                    st.write(f"Reading **{result['title']}**  \n{result['url']}")
                    
                    #to_scrape = check_page_to_scrape(result['url'], result['content'], user_query)
                    #print(to_scrape)
                    if 1: #to_scrape:
                        # Process content
                        scraped = scrape_page(result['url'])
                        if scraped and scraped['content']:
                            st.session_state.sources.append(scraped['url'])
                            learning = analyze_content(scraped['content'], user_query)
                            st.info(f"learning of page {scraped['url']} \n {learning}")
                            st.session_state.learnings.append(learning)
                    else:
                        st.info(f"skipping page {result['url']}")
            
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
            
            # Reset session state
            # keys_to_reset = ['learnings', 'sources', 'depth', 'clarified']
            # for key in keys_to_reset:
            #     if key in st.session_state:
            #         del st.session_state[key]

if __name__ == "__main__":
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Main", "Settings"])
    
    if page == "Main":
        main_app()
    elif page == "Settings":
        settings_page()


## run with streamlit run app.py   --server.headless true
