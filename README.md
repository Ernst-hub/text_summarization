# Open source Llama text summarizer project

## Introduction
The following repository contains a collected set of files that are used to create a text summarizer.

Simply provide a URL to a text you want summarized and a question you want answered and the model will do the rest.

Example:

![example](/assets/screenshot.png)

*Reliance on third party services*:

The primary model used is Ollama taken from the following site: [Ollama](https://ollama.ai/).
Furthermore, the repository uses [LangChain](https://www.langchain.com) to serve the model and a web interface for the user is provided via [Streamlit](https://www.streamlit.io/).

## Requirements
A CPU, the model is quantized and optimized for CPU, Metal, and CUDA by the awesome team behind [ggml](https://github.com/ggerganov/ggml)

## Installation
The following installation steps are required to run the project:
1. Clone the repository
2. Install the requirements
3. Setup [Ollama](https://ollama.ai/)
   1. Install Ollama
   2. Pull the model: "llama2-7b" (chat)
   3. Use the model file to create a QA model called "summarizev2"
4. Either run the model via the command line or via the web interface

## Usage

**command line:**
locate the file `main.py` and run it via the command line.

Example usage: `python main.py --url="https://en.wikipedia.org/wiki/Francisco_Goya" --question="Who was Goya?"`

*Arguments*

Use `python main.py --help` to see the full list of arguments:
- `--url` - the url of the text to summarize
- `--question` - the question to ask the model
- `--model` - the model to use for summarization
- `--base-url` - the base url for Ollama
- `--verbose` - if True, print out debug information
- `--chunk-size` - size of chunks to split text into
- `--embedding-model` - embedding model to use
- `--retriever` - retriever to use
- `--device` - device to use

**web interface:**
locate the file `streamlit_app.py` and run it via the command line.

Example usage: `streamlit run streamlit_app.py`. This should result in a web interface being opened in your browser. It should look similar to the image provided above.
