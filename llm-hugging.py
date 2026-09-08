import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch


load_dotenv(
    dotenv_path=r"D:\ML Projects\machine_learning_projects\.env"
)
DOC_PATH = r"D:\ML Projects\machine_learning_projects\iThesis-complete.pdf"
CHROMA_PATH = "lmp-thesis-rag"


# Load PDF
print("Loading PDF...")
loader = PyPDFLoader(DOC_PATH)
pages = loader.load()
print(f"Pages loaded: {len(pages)}")


# Split document into chunks
print("Splitting document...")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(pages)
print(f"Chunks created: {len(chunks)}")


# Create embeddings
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# Create Chroma vector database
print("Creating vector database...")
db_chroma = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_PATH
)
print("Vector database ready")


# Prompt template
PROMPT_TEMPLATE = """

Answer the question using only the information provided.

Context:
{context}

Question:
{question}

Rules:
- Do not add outside information.
- Give a detailed and precise answer.
- Do not mention "according to the context".
"""

prompt_template = ChatPromptTemplate.from_template(
    PROMPT_TEMPLATE
)


# Free local LLM (Hugging Face)
print("Loading local LLM ...")
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"

# 4-bit quantization keeps VRAM usage low (~5-6GB) while staying quality-preserving
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
hf_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

text_gen_pipeline = pipeline(
    "text-generation",
    model=hf_model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    temperature=0.3,
    do_sample=True,
    return_full_text=False,
)

model = HuggingFacePipeline(pipeline=text_gen_pipeline)
print("Model loaded.\n")


# Interactive Q&A loop
print("Ask questions about the document (type 'exit' to quit).\n")
while True:
    query = input("Your question: ")
    if query.strip().lower() in ("exit", "quit"):
        print("Goodbye!")
        break

    if not query.strip():
        continue

    # Retrieve relevant chunks
    print("Searching documents...")
    docs = db_chroma.similarity_search_with_score(
        query,
        k=15
    )

    context_text = "\n\n".join(
        [
            doc.page_content
            for doc, score in docs
        ]
    )

    print("\n--- Retrieved chunks ---")
    for i, (doc, score) in enumerate(docs):
        print(f"[{i}] score={score:.3f} page={doc.metadata.get('page')}")
        print(doc.page_content[:150], "...\n")
    print("--- end chunks ---\n")

    # Build prompt
    prompt = prompt_template.format(
        context=context_text,
        question=query
    )

    # Generate answer
    print("Generating answer...")
    response = model.invoke(prompt)
    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print(response)
    print()