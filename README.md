# pyChercheur
Trial of a AI deep research tool

How it works:
* you ask a question
* it asks you up to 3 questions to have a better understanding of your question
* it generates `SEARCH_WIDTH` search queries
* it chooses a maximum of `MAX_PAGE_PER_LEVEL` pages to read
* it scrapes it and analyses their content
* it loops `SEARCH_DEPTH` times

## Usage

### Requirements

* Firecrawl API key or a self-hosted instance: `FIRECRAWL_API_KEY` and `FIRECRAWL_ENDPOINT`
* SearxNG instance: `SEARXNG_INSTANCE`
* LLM API key. It uses LiteLLM of basically the one of your choice.
* uv

### Run

run with: `uv run streamlit run app.py`


## Ideas for the future
* scrape pages in batch to improve speed
* improve the prompts for the LLM. The current one are inspired from https://github.com/dzhng/deep-research