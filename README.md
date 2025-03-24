# RAGent

A Retrieval Augmented Generation (RAG) pipeline using Azure OpenAI and LangChain.

## Overview

RAGent is a command-line tool that enables:
- Ingesting and processing documents into a vector database
- Interactive chat with a large language model (LLM) using your documents as context

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ragent.git
cd ragent

# Install the package
pip install -e .
```

## Configuration

Create a `.env` file in the root directory with your Azure OpenAI credentials and settings. You can use the provided `.env.example` as a template:

```bash
cp .env.example .env
```

Then edit the `.env` file with your Azure OpenAI credentials:

```
# Azure OpenAI Configuration
RAGENT_AOAI_ENDPOINT="https://your-api-endpoint.openai.azure.com/"
RAGENT_EMBEDDING_DEPLOYMENT_NAME=text-embedding-3-large
RAGENT_EMBEDDING_DEPLOYMENT_VERSION=2025-01-01-preview
RAGENT_LLM_DEPLOYMENT_NAME=gpt-4
RAGENT_LLM_DEPLOYMENT_VERSION=2025-01-01-preview

# Document Processing Configuration
RAGENT_CHUNK_SIZE=500
RAGENT_CHUNK_OVERLAP=50
RAGENT_BATCH_SIZE=100

# Vector Store Configuration
RAGENT_PERSIST_DIR=./chroma_db
```

## Usage

RAGent provides two main commands:

### 1. Ingest Documents

Process and store documents in the vector database:

```bash
ragent ingest <directory_path>
```

### 2. Chat with Your Documents

Start an interactive chat session with the LLM using your ingested documents as context:

```bash
ragent chat
```

## Requirements

- Python 3.9 or higher
- Azure OpenAI API access

## Dependencies

- LangChain
- ChromaDB
- Click
- Python-dotenv

## License

[MIT](LICENSE) 