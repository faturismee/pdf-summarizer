from medium, read with freedium.

Building a Custom Summarization App with Streamlit and LangChain
Building a simple app to create custom summaries out of any pdf
Lucas Soares
Lucas Soares

Follow
MLearning.ai
MLearning.ai

androidstudio
~5 min read
·
June 14, 2023 (Updated: June 16, 2023)
·
Free: No
None
Photo by James Harrison on Unsplash
How to Build a Custom Summarization App with LangChain and Streamlit
One of my favorite applications of modern Large Language Models is to create summaries of PDFs.

More than just any summary, I want the ability to create customizable summaries that can fit any research or learning need.

In this article, you'll learn how to build a simple custom summarization app, using LangChain and Streamlit.

If you prefer video, check out my Youtube video on this topic here:


Steps to Build a Summarization App with Custom Prompts
This app will allow users to create custom prompts to summarize PDF files using AI-powered language models like ChatGPT and GPT-4. The idea is to provide an interface for creating custom summaries out of any PDF file.

Steps

Import dependencies
Define the helper functions
Create a responsive user interface with Streamlit.
Running the App
In the sections below, let's walk you through the code.

Importing dependencies
Copy
import openai
import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

# Make sure you have the OPENAI API KEY set in your environment
openai.api_key = os.environ['OPENAI_API_KEY']
In this section, we import the required modules and libraries for implementing the app. They include openai'sGPT models, streamlit for the user interface, and some custom classes and functions for processing text using langchain

Define the helper functions
Copy
@st.cache_data
def setup_documents(pdf_file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(pdf_file_path)
    docs_raw = loader.load()
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents(docs_raw_text)
    return docs
The setup_documents function is responsible for loading a PDF file, extracting text from it, and then splitting the text into smaller chunks based on the given chunk size and overlap.

We use the PyPDFLoader class to load the PDF file and obtain its content. The RecursiveCharacterTextSplitter class is then used to split the text into smaller chunks.

Copy
def custom_summary(docs,llm, custom_prompt, chain_type, num_summaries):
    custom_prompt = custom_prompt + """:\n\n {text}"""
    COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables=["text"])
    MAP_PROMPT = PromptTemplate(template="Summarize:\n\n{text}", input_variables=["text"])
    if chain_type == "map_reduce":
        chain = load_summarize_chain(llm, chain_type=chain_type, 
                                    map_prompt=MAP_PROMPT, combine_prompt=COMBINE_PROMPT)
    else:
        chain = load_summarize_chain(llm, chain_type=chain_type)
    summaries = []
    for i in range(num_summaries):
        summary_output = chain({"input_documents": docs}, return_only_outputs=True)["output_text"]
        summaries.append(summary_output)
    
    return summaries
The custom_summaryfunction takes the documents (chunks of text), language model (llm), custom summary prompt, chain type, and the number of summaries the user wants.

It then creates a summarization chain based on the specified chain type and the language model. The function subsequently loops through the documents, generating summaries based on the given chain.

Copy
@st.cache_data
def color_chunks(text: str, chunk_size: int, overlap_size: int) -> str:
    overlap_color = "#808080" # Light gray for the overlap
    chunk_colors = ["#a8d08d", "#c6dbef", "#e6550d", "#fd8d3c", "#fdae6b", "#fdd0a2"] # Different shades of green for chunks

    colored_text = ""
    overlap = ""
    color_index = 0

    for i in range(0, len(text), chunk_size-overlap_size):
        chunk = text[i:i+chunk_size]
        if overlap:
            colored_text += f'<mark style="background-color: {overlap_color};">{overlap}</mark>'
        chunk = chunk[len(overlap):]
        colored_text += f'<mark style="background-color: {chunk_colors[color_index]};">{chunk}</mark>'
        color_index = (color_index + 1) % len(chunk_colors)
        overlap = text[i+chunk_size-overlap_size:i+chunk_size]

    return colored_text
The color_chunks function is responsible for creating a visually appealing HTML representation of text chunks with overlaps. This function will be useful for debugging chunk size and overlap when visualizing how the text will be split.

Create a responsive user interface with Streamlit.
Copy
def main():
    st.set_page_config(layout="wide")
    st.title("Custom Summarization App")
    chain_type = st.sidebar.selectbox("Chain Type", ["map_reduce", "stuff", "refine"])
    chunk_size = st.sidebar.slider("Chunk Size", min_value=100, max_value=10000, step=100, value=1900)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=100, max_value=10000, step=100, value=200)
    
    if st.sidebar.checkbox("Debug chunk size"):
        st.header("Interactive Text Chunk Visualization")

        text_input = st.text_area("Input Text", "This is a test text to showcase the functionality of the interactive text chunk visualizer.")

        # Set the minimum to 1, the maximum to 5000 and default to 100
        html_code = color_chunks(text_input, chunk_size, chunk_overlap)
        st.markdown(html_code, unsafe_allow_html=True)
    
    else:
        user_prompt = st.text_input("Enter the user prompt")
        pdf_file_path = st.text_input("Enter the pdf file path")
        
        temperature = st.sidebar.number_input("ChatGPT Temperature", min_value=0.0, max_value=1.0, step=0.1, value=0.0)
        num_summaries = st.sidebar.number_input("Number of Summaries", min_value=1, max_value=10, step=1, value=1)
        
        # make the choice of llm to select from a selectbox
        llm = st.sidebar.selectbox("LLM", ["ChatGPT", "GPT4", ""])
        if llm == "ChatGPT":
            llm = ChatOpenAI(temperature=temperature)
        elif llm == "GPT4":
            llm = ChatOpenAI(model_name="gpt-4",temperature=temperature)
        
        if pdf_file_path != "":
            docs = setup_documents(pdf_file_path, chunk_size, chunk_overlap)
            st.write("Pdf was loaded successfully")
            
            if st.button("Summarize"):
                result = custom_summary(docs,llm, user_prompt, chain_type, num_summaries)
                st.write("Summaries:")
                for summary in result:
                    st.write(summary)

if __name__ == "__main__":
    main()
In the main() function, we implement the user interface of the app using Streamlit. We set the page configuration, create titles, and provide options for users to select the language model, chain type, chunk size, and chunk overlap values.

Based on the inputs, the app either displays the interactive text chunk visualizer when the user enables the "Debug chunk size" option or generates a custom summary from a PDF file using the user-selected language model and the custom prompt.

Running the App
Now, we can run the app with

Copy
streamlit run custom_summarization_app.py
You should see an interface that looks something like this:

None
Image by the author. UI Interface for the summarization app.
Now I can add a custom prompt and a path to a pdf:

None
Image by the author
Finally, I can click the Summarize button to get the summary:

None
Image by the author
And if you click the debug chunk size option and add some text:

None
Image by the author
Here you have the option of adjusting the chunk size and overlap options and interactively visualize how the text will be divided.

You can check out the full source code on the repo I created here:

GitHub - EnkrateiaLucca/summarization_with_langchain: Using langchain for summarization
Using langchain for summarization. Contribute to EnkrateiaLucca/summarization_with_langchain development by creating an…
github.com

Conclusion
The custom summarization app using OpenAI and Streamlit provides a powerful way to generate AI-based summaries from PDF files with custom prompts and different language models.

I want to add some options in the future to create summaries for multiple papers as well as incorporate additional features like automatic keyword extraction, supporting various input file formats, and improving summarization quality with feedback loops.

If you liked this post, join Medium, subscribe to my Youtube channel and my newsletter. Thanks and see you next time! Cheers! :)