import click
from pathlib import Path
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from ragent.graph.rag_graph import create_rag_graph

def chat_with_documents(
    persist_dir: Path,
    aoai_endpoint: str,
    embedding_deployment_name: str,
    embedding_deployment_version: str,
    llm_deployment_name: str,
    llm_deployment_version: str,
) -> None:
    """Chat with documents using the RAG graph."""
    token_provider = get_bearer_token_provider(
        DefaultAzureCredential(),
        "https://cognitiveservices.azure.com/.default"
    )

    # Initialize embeddings
    embedding_model = AzureOpenAIEmbeddings(
        azure_endpoint=aoai_endpoint,
        deployment=embedding_deployment_name,
        openai_api_version=embedding_deployment_version,
        azure_ad_token_provider=token_provider
    )

    # Load vector store
    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding_model
    )

    # Initialize LLM
    llm = AzureChatOpenAI(
        azure_endpoint=aoai_endpoint,
        deployment_name=llm_deployment_name,
        azure_ad_token_provider=token_provider,
        api_version=llm_deployment_version
    )

    # Create and compile the graph
    graph = create_rag_graph(llm=llm, vectorstore=vectorstore)

    # Run the chat loop
    system_message = SystemMessage(
        content="""You are a helpful AI assistant that answers questions based on provided documents. 
            Use the search_documents tool to find relevant information. 
            Always cite your sources when providing information from documents."""
    )
    config = {"configurable": {"thread_id": "1"}}
    messages = {"messages": [system_message]}

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            break
            
        messages = graph.invoke(
            {"messages": [HumanMessage(content=user_input)]}, 
            config
        )
        
        print("\nAssistant:")
        print(messages["messages"][-1].content)

@click.command()
@click.argument("persist_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), envvar="RAGENT_PERSIST_DIR")
@click.option("--aoai-endpoint", envvar="RAGENT_AOAI_ENDPOINT", required=True, help="Azure OpenAI endpoint")
@click.option("--embedding-deployment-name", envvar="RAGENT_EMBEDDING_DEPLOYMENT_NAME", required=True, help="Azure OpenAI embedding deployment name")
@click.option("--embedding-deployment-version", envvar="RAGENT_EMBEDDING_DEPLOYMENT_VERSION", required=True, help="Azure OpenAI embedding deployment version")
@click.option("--llm-deployment-name", envvar="RAGENT_LLM_DEPLOYMENT_NAME", required=True, help="Azure OpenAI LLM deployment name")
@click.option("--llm-deployment-version", envvar="RAGENT_LLM_DEPLOYMENT_VERSION", required=True, help="Azure OpenAI LLM deployment version")
def chat(
    persist_dir: Path,
    aoai_endpoint: str,
    embedding_deployment_name: str,
    embedding_deployment_version: str,
    llm_deployment_name: str,
    llm_deployment_version: str,
) -> None:
    """Chat with documents using the RAG graph."""
    chat_with_documents(
        persist_dir=persist_dir,
        aoai_endpoint=aoai_endpoint,
        embedding_deployment_name=embedding_deployment_name,
        embedding_deployment_version=embedding_deployment_version,
        llm_deployment_name=llm_deployment_name,
        llm_deployment_version=llm_deployment_version,
    ) 