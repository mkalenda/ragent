import click
from pathlib import Path
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragent.loader.document_loader import load_documents
from tqdm import tqdm

def ingest_documents(
    input_dir: Path,
    persist_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_deployment_name: str,
    embedding_deployment_version: str,
    aoai_endpoint: str,
    batch_size: int = 100
) -> None:
    """Ingest documents from the input directory into the vector store."""
    # Initialize embeddings
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )
    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint=aoai_endpoint,
        deployment=embedding_deployment_name,
        openai_api_version=embedding_deployment_version,
        azure_ad_token_provider=token_provider
    )

    # First count total files to load
    total_files = sum(1 for _ in Path(input_dir).rglob('*') if _.is_file())
    with tqdm(total=total_files, desc="Loading files", unit="file") as pbar:
        documents = load_documents(input_dir, progress_bar=pbar)

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    
    if not chunks:
        print("No documents found to ingest.")
        return
        
    print(f"Processing {len(chunks)} chunks...")
    # Create vector store with initial batch
    initial_batch = chunks[:batch_size]
    vectorstore = Chroma.from_documents(
        documents=initial_batch,
        embedding=embedding_model,
        persist_directory=str(persist_dir)
    )
    
    # Process remaining documents in batches
    if len(chunks) > batch_size:
        remaining_chunks = chunks[batch_size:]
        for batch in tqdm(
            [remaining_chunks[i:i + batch_size] for i in range(0, len(remaining_chunks), batch_size)],
            desc="Processing batches",
            unit="batch"
        ):
            if batch:  # Only add non-empty batches
                vectorstore.add_documents(batch)
    
    print("Ingestion complete!")

@click.command()
@click.argument('input_dir', type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option('--chunk-size', default=500, help='Size of text chunks for splitting documents', envvar='RAGENT_CHUNK_SIZE')
@click.option('--chunk-overlap', default=50, help='Overlap between chunks', envvar='RAGENT_CHUNK_OVERLAP')
@click.option('--persist-dir', default='./chroma_db', help='Directory to persist the vector store', envvar='RAGENT_PERSIST_DIR')
@click.option('--aoai-endpoint', required=True, help='Azure OpenAI endpoint', envvar='RAGENT_AOAI_ENDPOINT')
@click.option('--embedding-deployment-name', required=True, help='Azure OpenAI embedding deployment name', envvar='RAGENT_EMBEDDING_DEPLOYMENT_NAME')
@click.option('--embedding-deployment-version', required=True, help='Azure OpenAI embedding deployment version', envvar='RAGENT_EMBEDDING_DEPLOYMENT_VERSION')
@click.option('--batch-size', default=100, help='Number of documents to process in each batch', envvar='RAGENT_BATCH_SIZE')
def ingest(
    input_dir: Path,
    persist_dir: Path,
    aoai_endpoint: str,
    embedding_deployment_name: str,
    embedding_deployment_version: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
) -> None:
    """Ingest documents from the input directory into the vector store."""
    ingest_documents(
        input_dir=input_dir,
        persist_dir=persist_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_deployment_name=embedding_deployment_name,
        embedding_deployment_version=embedding_deployment_version,
        aoai_endpoint=aoai_endpoint,
        batch_size=batch_size,
    ) 