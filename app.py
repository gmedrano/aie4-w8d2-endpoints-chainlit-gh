import os
import chainlit as cl
from dotenv import load_dotenv
from operator import itemgetter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.prompts import PromptTemplate
from tqdm.asyncio import tqdm_asyncio
import asyncio
from tqdm.asyncio import tqdm

# ---- Load environment variables from the .env file ---- #
"""
This function loads environment variables from the .env file
which contains sensitive information like API tokens and endpoint URLs.
It ensures those values can be accessed through `os.environ`.
"""
load_dotenv()

# ---- Fetch environment variables for Hugging Face API credentials ---- #
"""
We retrieve the following environment variables:
HF_LLM_ENDPOINT: The Hugging Face Inference endpoint for the LLM model.
HF_EMBED_ENDPOINT: The Hugging Face endpoint for the embedding model.
HF_TOKEN: The Hugging Face API token used to authenticate requests.
"""
HF_LLM_ENDPOINT = os.environ["HF_LLM_ENDPOINT"]
HF_EMBED_ENDPOINT = os.environ["HF_EMBED_ENDPOINT"]
HF_TOKEN = os.environ["HF_TOKEN"]

# ---- Load and process documents for retrieval ---- #
"""
1. Load the Paul Graham essays from a local text file using TextLoader.
2. Split the text into chunks of 1000 characters each with 30 characters of overlap between chunks.
   This is necessary to process large documents effectively during retrieval.
"""
document_loader = TextLoader("./data/paul_graham_essays.txt")  # Load documents from text file
documents = document_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)  # Split the document into manageable chunks
split_documents = text_splitter.split_documents(documents)

# ---- Create embeddings using the Hugging Face Inference API ---- #
"""
This block sets up the embedding model to convert text chunks into vector representations.
These embeddings are essential for vector-based retrieval, enabling similarity search.
"""
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model=HF_EMBED_ENDPOINT,
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN,
)

# ---- Asynchronous document indexing and vector store creation ---- #
"""
These functions handle adding document embeddings to the FAISS vector store in batches.
We process batches asynchronously to optimize time spent on larger datasets.
"""
async def add_documents_async(vectorstore, documents):
    """Adds a batch of documents asynchronously to the FAISS vectorstore."""
    await vectorstore.aadd_documents(documents)

async def process_batch(vectorstore, batch, is_first_batch, pbar):
    """
    Processes each batch of documents. If it's the first batch, it initializes the vectorstore.
    Subsequent batches are added to the existing vectorstore.
    """
    if is_first_batch:
        result = await FAISS.afrom_documents(batch, hf_embeddings)
    else:
        await add_documents_async(vectorstore, batch)
        result = vectorstore
    pbar.update(len(batch))
    return result

async def main():
    """Main function to handle document indexing across batches and return the retriever."""
    print("Indexing Files")
    
    vectorstore = None  # Initialize the vectorstore
    batch_size = 32  # Set the batch size for document processing
    
    # Split documents into batches of 32
    batches = [split_documents[i:i+batch_size] for i in range(0, len(split_documents), batch_size)]
    
    # Process all batches asynchronously
    async def process_all_batches():
        nonlocal vectorstore
        tasks = []
        pbars = []
        
        for i, batch in enumerate(batches):
            pbar = tqdm(total=len(batch), desc=f"Batch {i+1}/{len(batches)}", position=i)
            pbars.append(pbar)
            
            # Process the first batch and initialize the vectorstore
            if i == 0:
                vectorstore = await process_batch(None, batch, True, pbar)
            else:
                tasks.append(process_batch(vectorstore, batch, False, pbar))
        
        # Wait for all batches to finish processing
        if tasks:
            await asyncio.gather(*tasks)
        
        for pbar in pbars:
            pbar.close()
    
    await process_all_batches()
    
    # Return the retriever object that can be used to search the indexed documents
    hf_retriever = vectorstore.as_retriever()
    print("\nIndexing complete. Vectorstore is ready for use.")
    return hf_retriever

# ---- Run the indexing and retrieve the retriever ---- #
"""
This runs the main indexing process and retrieves the `hf_retriever`,
which will be used in our RAG (Retrieval Augmented Generation) pipeline.
"""
async def run():
    retriever = await main()
    return retriever

hf_retriever = asyncio.run(run())

# ---- Create a prompt template for generating LLM responses ---- #
"""
Define a RAG (Retrieval Augmented Generation) prompt template.
The template includes placeholders for a user query and the context retrieved from documents.
"""
RAG_PROMPT_TEMPLATE = """\
<|start_header_id|>system<|end_header_id|>
You are a helpful assistant. You answer user questions based on provided context. If you can't answer the question with the provided context, say you don't know.<|eot_id|>

<|start_header_id|>user<|end_header_id|>
User Query:
{query}

Context:
{context}<|eot_id|>

<|start_header_id|>assistant<|end_header_id|>
"""

rag_prompt = PromptTemplate.from_template(RAG_PROMPT_TEMPLATE)  # Create prompt template instance

# ---- Set up Hugging Face LLM endpoint for generation ---- #
"""
This configures the Hugging Face LLM (language model) to generate responses to user queries.
We define several parameters like the number of tokens, top_k, temperature, and repetition penalty
to control how the model generates text.
"""
hf_llm = HuggingFaceEndpoint(
    endpoint_url=HF_LLM_ENDPOINT,
    max_new_tokens=512,
    top_k=10,
    top_p=0.95,
    temperature=0.3,
    repetition_penalty=1.15,
    huggingfacehub_api_token=HF_TOKEN,
)

# ---- Rename the assistant in the chat interface ---- #
"""
This renames the author of the assistant's responses in the chat interface
to 'Paul Graham Essay Bot' instead of the default 'Assistant'.
"""
@cl.author_rename
def rename(original_author: str):
    rename_dict = {
        "Assistant" : "Paul Graham Essay Bot"
    }
    return rename_dict.get(original_author, original_author)

# ---- Handle the start of a new chat session ---- #
"""
This function initializes the RAG chain when a user starts a new chat session.
The RAG chain is stored in the user session, which ensures each session gets its own chain.
"""
@cl.on_chat_start
async def start_chat():
    lcel_rag_chain = (
        {"context": itemgetter("query") | hf_retriever, "query": itemgetter("query")}
        | rag_prompt | hf_llm
    )
    
    # Store the RAG chain in the user session
    cl.user_session.set("lcel_rag_chain", lcel_rag_chain)

# ---- Handle incoming messages from users ---- #
"""
This function handles incoming messages from the user.
It uses the RAG chain to retrieve relevant context, generate a response, and stream it back to the user.
"""
@cl.on_message  
async def main(message: cl.Message):
    """
    This function will be called every time a message is received from a session.

    We will use the LCEL RAG chain to generate a response to the user query.

    The LCEL RAG chain is stored in the user session, and is unique to each user session - this is why we can access it here.
    """
    lcel_rag_chain = cl.user_session.get("lcel_rag_chain")

    msg = cl.Message(content="")

    # Use make_async and stream the response chunks without LangchainCallbackHandler
    for chunk in await cl.make_async(lcel_rag_chain.stream)(
        {"query": message.content},
        # Remove the LangchainCallbackHandler dependency
        # Use direct async message streaming
    ):
        await msg.stream_token(chunk)

    # Send the final message back to the user
    await msg.send()
