import os
from pathlib import Path
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


load_dotenv()

PDF_PATH = Path(__file__).parent.parent / "data"
CHROMA_DIR   = "chroma_db"
CHUNK_SIZE   = 512
CHUNK_OVERLAP = 64
STORE_MARKDOWN = True

def load_documents():
    docs = []

    for pdf_path in PDF_PATH.glob("*.pdf"):
        markdown_path = pdf_path.with_suffix(".md")

        if markdown_path.exists():
            print(f"Using cached markdown for {pdf_path.name}...")
            markdown = markdown_path.read_text()
        else:
            print(f"Parsing {pdf_path.name} with Docling...")
            converter = DocumentConverter()
            markdown = converter.convert(str(pdf_path)).document.export_to_markdown()
            markdown_path.write_text(markdown)

        docs.append(Document(
            page_content=markdown,
            metadata={"source": pdf_path.name}
        ))

    print(f"\nLoaded {len(docs)} document(s)")
    return docs


def split_documents(docs):
    splitter=RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks=splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks

def build_vectorstore(chunks):
    print("\n Embedding and storing in Chroma...")
    Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=CHROMA_DIR)
    print(f"Done. Vectorstore saved to {CHROMA_DIR}/")

if __name__ == "__main__":
    docs   = load_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("\n✅ Ingestion complete.")
