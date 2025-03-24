from typing import List
from langchain_openai import AzureChatOpenAI
from langchain_chroma import Chroma
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
from langchain.tools import tool
from langgraph.checkpoint.memory import MemorySaver

def create_rag_graph(
    llm: AzureChatOpenAI,
    vectorstore: Chroma
) -> StateGraph:
    """Create a RAG graph with the given LLM and vectorstore."""
    
    @tool
    def search_documents(query: str) -> str:
        """Search the vector store for relevant documents to answer the query."""
        docs = vectorstore.similarity_search(query, k=20)
        return "\n\n".join([f"Document {i+1}:\n{doc.page_content} \n\n Source: {doc.metadata['source']} \n\n Metadata: {doc.metadata}" for i, doc in enumerate(docs)])

    # Create tools and tool node
    tools = [search_documents]
    tool_node = ToolNode(tools)

    # Bind tools to the model
    model_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        """Call the model with the current messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    # Define the graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("assistant", assistant)
    builder.add_node("tools", tool_node)

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    return builder.compile(checkpointer=MemorySaver()) 