# Paul Graham Essay Bot - Chainlit Application

Welcome to the **Paul Graham Essay Bot**! This Chainlit-powered application uses a Retrieval Augmented Generation (RAG) pipeline to answer questions based on the content of Paul Graham's essays.

## Features
- **Document Retrieval**: Uses FAISS vector store to index and retrieve relevant chunks from Paul Graham's essays.
- **Language Generation**: Utilizes Hugging Face's Large Language Models (LLMs) to generate responses based on retrieved content.
- **Interactive Chat**: Chainlit provides a chat interface for interacting with the RAG pipeline.

## Prerequisites

Before you can run this app, make sure you have the following tools installed:

- **Python 3.8+**
- **Chainlit** (`pip install chainlit`)
- **Hugging Face API Token** (from [Hugging Face](https://huggingface.co/))

Make sure the following environment variables are set in your `.env` file:
- `HF_LLM_ENDPOINT`: Your Hugging Face LLM endpoint
- `HF_EMBED_ENDPOINT`: Your Hugging Face embedding endpoint
- `HF_TOKEN`: Your Hugging Face API token