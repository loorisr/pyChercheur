import streamlit as st
import requests
import litellm
import re
from firecrawl.firecrawl import FirecrawlApp
from dotenv import load_dotenv
import os
from datetime import datetime
from fpdf import FPDF  # Import the FPDF library

# Load environment variables
load_dotenv()

# Configuration
SEARXNG_INSTANCE = os.getenv('SEARXNG_INSTANCE')
FIRECRAWL_API_KEY = os.getenv('FIRECRAWL_API_KEY')
FIRECRAWL_ENDPOINT = os.getenv('FIRECRAWL_ENDPOINT')
SEARCH_WIDTH = int(os.getenv('SEARCH_WIDTH', 3))
SEARCH_DEPTH = int(os.getenv('SEARCH_DEPTH', 3))
MAX_PAGE_PER_LEVEL = int(os.getenv('MAX_PAGE_PER_LEVEL', 5))

# Initialize APIs
firecrawl = FirecrawlApp(api_key=FIRECRAWL_API_KEY, api_url=FIRECRAWL_ENDPOINT)


def system_prompt():
    now = datetime.now().isoformat()
    return f"""You are an expert researcher. Today is {now}. Follow these instructions when responding:
  - You may be asked to research subjects that is after your knowledge cutoff, assume the user is right when presented with news.
  - The user is a highly experienced analyst, no need to simplify it, be as detailed as possible and make sure your response is correct.
  - Be highly organized.
  - Suggest solutions that I didn't think about.
  - Be proactive and anticipate my needs.
  - Treat me as an expert in all subject matter.
  - Mistakes erode my trust, so be accurate and thorough.
  - Provide detailed explanations, I'm comfortable with lots of detail.
  - Value good arguments over authorities, the source is irrelevant.
  - Consider new technologies and contrarian ideas, not just the conventional wisdom.
  - You may use high levels of speculation or prediction, just flag it for me."""

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

def generate_llm_response(user_prompt, system_prompt=None, task_type="reasoning"):
    """Generate response using LiteLLM."""
    model_name = st.session_state.reasoning_model if task_type == "reasoning" else st.session_state.summarization_model
    model_name = st.session_state.llm_provider + "/" + model_name
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

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
    Given the following query from the user, ask some follow up questions
    to clarify the research direction. Return a maximum of 3 questions,
    but feel free to return less if the original query is clear: <query>{original_query}</query>`,
    """
    return generate_llm_response(clarification_prompt, system_prompt=system_prompt(), task_type="reasoning")

def generate_search_queries(research_context):
    prompt = f"""
    Given the following prompt from the user, generate a list of SERP queries
    to research the topic. Return a maximum of {SEARCH_WIDTH} queries,
    but feel free to return less if the original prompt is clear.
    Make sure each query is unique and not similar to each other: <prompt>{research_context}</prompt>
    Return each query on a new line without numbering.
    """
    search_queries = generate_llm_response(prompt, system_prompt=system_prompt(), task_type="reasoning")
    return search_queries.splitlines()

# def check_page_to_scrape(url, content, research_goal):  ################
#     prompt = f"""
#     Analyze this url {url} in relation to: {research_goal}.
#     The summury of this url is {content}.
#     Tell by true or false if it would be interesting to get the complete page or if we can skip it.
    
#     Respond only with true or false.
#     """
#     return generate_llm_response(prompt, task_type="reasoning")


def list_pages_to_scrape(search_results, research_goal):  ################
    prompt = f"""
    Analyze theses searches results (url and content) in relation to: {research_goal}.
    Select only a maximum of {MAX_PAGE_PER_LEVEL} urls that are the most relevant to answer to the research.

    <search_results>{search_results}</search_results>
    
    Return each url that you have selected on a new line without numbering.
    """
    return generate_llm_response(prompt, task_type="reasoning")

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
        results = response.json()["results"]
        search_results = []
        for item in results:
            search_results.append({"title": item["title"], "content": item["content"], "url": item["url"]})
            
        #print(search_results)
        return search_results
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
    Provide several learnings from this content that are relevant to 
    Make sure each learning is unique and not similar to each other.
    The learnings should be concise and to the point, as detailed and information dense as possible.
    Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any exact metrics, numbers, or dates. 
    
    <content>{content[:12000]}</content>
    """
    return generate_llm_response(prompt, task_type="summarization")

def generate_report(learnings, sources, research_goal):
    prompt = f"""
    Given the following prompt from the user, write a final report on the topic
    using the learnings from research. Make it as as detailed as possible, aim for 3 or more pages,
    include ALL the learnings from research:
    
    <prompt>{research_goal}</prompt>
    
    Here are all the learnings from previous research:
    
    <learnings>{"\n".join(learnings)}</learnings>`.
    
    Write the report using markdown and add citation of your sources.

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
        index=1
    )
    
    st.subheader("Model Selection")
    st.selectbox(
        "Reasoning Model",
        ["gpt-4", "llama3-70b", "gemini-1.5-pro", "gemini-2.0-pro-exp-02-05", "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash"],
        key="reasoning_model",
        index=4
        )
    st.selectbox(
        "Summarization Model",
        ["gemma2:2b", "llama3-8b", "gemini-2.0-flash", "gemini-2.0-flash-lite-preview-02-05"],
        key="summarization_model",
        index=2
    )
    
    
    st.subheader("API Key")
    api_key = st.text_input(f"{llm_provider.capitalize()} API Key", 
                          type="password",
                          key=f"{llm_provider}_key", value=os.getenv(f"{llm_provider.upper()}_API_KEY"))
    
    if api_key:
        os.environ[f"{llm_provider.upper()}_API_KEY"] = api_key
        
    
    global SEARCH_WIDTH, SEARCH_DEPTH, MAX_PAGE_PER_LEVEL
    st.subheader("Research settings")
    st.write("Here you can customize the search depth, width, and max pages per level of the research.")   

    SEARCH_WIDTH = st.number_input("Number of search queries per level", min_value=1, max_value=10, value=SEARCH_WIDTH, key="search_width_input")
    SEARCH_DEPTH = st.number_input("Number of search iterations", min_value=1, max_value=10, value=SEARCH_DEPTH, key="search_depth_input")
    MAX_PAGE_PER_LEVEL = st.number_input("Max number of pages to scrape per level", min_value=1, max_value=20, value=MAX_PAGE_PER_LEVEL, key="max_page_per_level_input")

def get_content_by_url(data_list, target_url):
    # Iterate through the list to find the matching URL
    for item in data_list:
        if item["url"] == target_url:
            return item
    # Return None if no match is found
    return None

def markdown_to_pdf(markdown_text, filename="report.pdf"):
    """Converts markdown text to a PDF file using FPDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for line in markdown_text.split("\n"):
        if line.startswith("#"):  # Handle headers
            if line.startswith("##"):
                pdf.set_font("Arial", 'B', size=14)
            elif line.startswith("###"):
                pdf.set_font("Arial", 'B', size=12)
            else:
                pdf.set_font("Arial", 'B', size=16)
            pdf.cell(200, 10, txt=line.replace("#", "").strip(), ln=True)
            pdf.set_font("Arial", size=12)
        elif line.startswith("["): #handle link like [1] url.com
          pdf.cell(200, 5, txt=line, ln=True)
        elif line.strip() == "":  # Add paragraph break
            pdf.cell(200, 5, txt="", ln=True)
        else:
            pdf.multi_cell(0, 5, txt=line)
        
    pdf.output(filename)

def main_app():
    st.title("Deep Research Assistant")
    user_query = st.text_input("Enter your research question:") # , value="comparaison of scraping API")

    if user_query:
        if "clarified" not in st.session_state:
            clarification = clarify_query(user_query)
            print("-"*20)
            print(f"clarification {clarification}")
            print("-"*20)
            st.info(clarification)
            clarification_answer = st.text_input("Please provide additional details:") #, value="list their price, free trial is available. return a comparative table.")
            if clarification_answer:
                user_query += f"\n {clarification_answer}"
                print("-"*20)
                print(f"user_query {user_query}")
                print("-"*20)
                st.session_state.clarified = True
                st.rerun()
            return

        if st.session_state.depth < SEARCH_DEPTH:
            print("-"*20)
            print(f"Search iteration #{st.session_state.depth}")
            st.write(f"## Search Iteration {st.session_state.depth + 1}")
            search_queries = generate_search_queries(user_query)
            search_queries = search_queries[:SEARCH_WIDTH]
            st.info(search_queries)
            print(f"Generated search queries {search_queries}")
            print("-"*20)
            
            search_results = []
            for query in search_queries:
                st.write(f"**Searching:** `{query}`")
                results = search_searxng(query)
                search_results.extend(results)
            
            #print(search_results)
            
            for result in search_results:
                #print(result)
                st.info(f"**{result['title']}**  \n{result['content']} \n {result['url']}")
            
            urls_to_scrape = list_pages_to_scrape(results, user_query)
            
            url_pattern = r'https?://[^\s]+'

            # Find all URLs in the string
            urls_to_scrape = re.findall(url_pattern, urls_to_scrape)

            print("-"*20)
            st.info(f"Selected urls to visit: {urls_to_scrape}")
            print(f"urls_to_scrape {urls_to_scrape}")
            print("-"*20)
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
            
            # Add a button to export the report to PDF
            if st.button("Export to PDF"):
                markdown_to_pdf(report)
                with open("report.pdf", "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name="report.pdf",
                        mime="application/pdf"
                    )
            
            print("-"*20)
            print("Done !")
            print("-"*20)
            
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
