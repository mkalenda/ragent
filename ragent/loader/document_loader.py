from pathlib import Path
import logging
from typing import List, Optional
from langchain_core.documents import Document
from tqdm import tqdm

from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain_community.document_loaders import UnstructuredXMLLoader

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Mapping of file extensions to loaders
loader_map = {
    '.txt': TextLoader,
    '.pdf': PyPDFLoader,
    '.docx': UnstructuredWordDocumentLoader,
    '.pptx': UnstructuredPowerPointLoader,
    '.md': UnstructuredMarkdownLoader,
    '.html': UnstructuredHTMLLoader,
    '.htm': UnstructuredHTMLLoader,
    '.csv': CSVLoader,
    '.json': lambda path: JSONLoader(path, jq_schema=".[]", text_content=False),
    '.epub': UnstructuredEPubLoader,
    '.xml': UnstructuredXMLLoader,
    '.eml': UnstructuredEmailLoader,
    '.msg': UnstructuredEmailLoader,
}

def load_documents(directory_path: str, recursive: bool = True, progress_bar: Optional[tqdm] = None) -> List[Document]:
    """Load documents from a directory and its subdirectories.
    
    Args:
        directory_path (str, optional): Path to the directory containing documents. Defaults to ".".
        recursive (bool, optional): Whether to recursively load documents from subdirectories. Defaults to True.
        progress_bar (Optional[tqdm], optional): Progress bar to update during loading. Defaults to None.
        
    Returns:
        List[Document]: List of loaded documents
    """
    loader = DirectoryLoader(
        path=directory_path,
        loader_cls=lambda path: _custom_loader(path, progress_bar),
        recursive=recursive,
        silent_errors=True
    )

    documents = loader.load()
    if progress_bar:
        progress_bar.close()
    logger.info(f"Successfully loaded {len(documents)} documents from {directory_path}")
    return documents

def _custom_loader(path: str, progress_bar: Optional[tqdm] = None) -> Document | None:
    """Custom loader that attempts to load files based on their extension.
    
    Args:
        path (str): Path to the file to load
        progress_bar (Optional[tqdm], optional): Progress bar to update. Defaults to None.
        
    Returns:
        Document | None: Loaded document or None if loading fails
    """
    ext = Path(path).suffix.lower()
    try:
        if ext in loader_map:
            doc = loader_map[ext](path)
            if progress_bar:
                progress_bar.update(1)
            return doc
        else:
            logger.warning(f"Unknown file extension {ext} for {path}, attempting to load as text file")
            doc = TextLoader(path)
            if progress_bar:
                progress_bar.update(1)
            return doc
    except Exception as e:
        logger.warning(f"Failed to load {path}: {str(e)}")
        if progress_bar:
            progress_bar.update(1)
        return None
