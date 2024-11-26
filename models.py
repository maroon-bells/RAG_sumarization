
import re
import PyPDF2 
import hashlib
from datetime import datetime
from typing import List, Optional, Tuple

from langchain_community.document_loaders import UnstructuredExcelLoader , PyPDFLoader
from pydantic import BaseModel
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
)
from unstructured.staging.huggingface import chunk_by_attention_window

from embeddings import EmbeddingModelSingleton

def process_pdf_text(file_path):
    arabic_pattern = r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        all_cleaned_text = []

        for page in reader.pages:
            text = page.extract_text()
            text = re.sub(arabic_pattern, '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
            text = re.sub(r'_+', '', text)
            text = re.sub(r'…+', '', text)
            text = re.sub(arabic_pattern, '', text)
            text = ' '.join(text.split())
            all_cleaned_text.append(text)

        return " ".join(all_cleaned_text)
    
class Data(BaseModel):
    """
    Represents a adding data  .

    Attributes:
        file_name (str): the name of the data
        content (str): Content of the news article (might contain HTML)
        added_ad (datetime): Date article was updated (RFC 3339)
    """

    data_name : str 
    added_at: datetime
    post_url: str

    def to_document(self) -> "Document":
        """
        Converts the news data to a Document object.

        Returns:
            Document: A Document object representing the new data.
        """

        document_id = hashlib.md5(self.data_name.encode()).hexdigest()
        document = Document(id=document_id)

        article_elements = self.extract_data()
        cleaned_content = clean_non_ascii_chars(
            replace_unicode_quotes(clean(article_elements))
        )
        document.text = [cleaned_content.replace("\n","")]
        document.metadata["data_name"] = self.data_name
        document.metadata["added_at"] = self.added_at
        document.metadata["post_url"] = self.post_url
        return document
    
    def extract_data(self):
        if self.data_name.endswith(".pdf"):
            loader = PyPDFLoader(self.data_name)
            text = process_pdf_text(self.data_name)
        elif self.data_name.endswith(".xlsx"):
            loader = UnstructuredExcelLoader(self.data_name)
            docs = loader.load()
            text = " ".join(doc.page_content for doc in docs)
        else:
            text = ""
        return text

class Document(BaseModel):
    """
    A Pydantic model representing a document.

    Attributes:
        id (str): The ID of the document.
        metadata (dict): The metadata of the document.
        text (str): The text of the document.
        chunks (list): The chunks of the document.
        embeddings (list): The embeddings of the document.

    Methods:
        to_payloads: Returns the payloads of the document.
        compute_chunks: Computes the chunks of the document.
        compute_embeddings: Computes the embeddings of the document.
    """

    id: str
    metadata: dict = {}
    text: list = [] 
    chunks: list = []
    embeddings: list = []

    def to_payloads(self) -> Tuple[List[str], List[dict]]:
        """
        Returns the payloads of the document.

        Returns:
            Tuple[List[str], List[dict]]: A tuple containing the IDs and payloads of the document.
        """

        payloads = []
        ids = []
        for chunk in self.chunks:
            payload = self.metadata
            payload.update({"text": chunk})
            # Create the chunk ID using the hash of the chunk to avoid storing duplicates.
            chunk_id = hashlib.md5(chunk.encode()).hexdigest()

            payloads.append(payload)
            ids.append(chunk_id)

        return ids, payloads
    
    def compute_chunks(self, model: EmbeddingModelSingleton) -> "Document":
        """
        Computes the chunks of the document.

        Args:
            model (EmbeddingModelSingleton): The embedding model to use for computing the chunks.

        Returns:
            Document: The document object with the computed chunks.
        """

        for item in self.text:
            chunked_item = chunk_by_attention_window(
                item, model.tokenizer, max_input_size=model.max_input_length
            )

            self.chunks.extend(chunked_item)
        return self
    
    def compute_embeddings(self, model: EmbeddingModelSingleton) -> "Document":
        """
        Computes the embeddings for each chunk in the document using the specified embedding model.

        Args:
            model (EmbeddingModelSingleton): The embedding model to use for computing the embeddings.

        Returns:
            Document: The document object with the computed embeddings.
        """

        for chunk in self.chunks:
            embedding = model(chunk, to_list=True)

            self.embeddings.append(embedding)

        return self
