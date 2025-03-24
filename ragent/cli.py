import click
from dotenv import load_dotenv
from pathlib import Path

from ragent.commands.ingest import ingest
from ragent.commands.chat import chat

# Load environment variables from .env file
load_dotenv()

@click.group()
def cli():
    """RAGent - A RAG pipeline using Azure OpenAI and LangChain."""
    pass

cli.add_command(ingest)
cli.add_command(chat)

if __name__ == "__main__":
    cli() 