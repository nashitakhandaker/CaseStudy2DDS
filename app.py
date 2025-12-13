import gradio as gr
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load PDF
reader = PdfReader("Abalone.pdf")
raw = "\n".join(page.extract_text() for page in reader.pages)

# Split chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " ", ""],
    length_function=len
)

chunks = splitter.split_text(raw)

# Create vector DB and embeddings
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

client = chromadb.Client()

collection = client.create_collection(
    name="abalone_db",
    metadata={"hnsw:space": "cosine"}
)

embeddings = embedder.encode(chunks, normalize_embeddings=True)

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[f"id_{i}" for i in range(len(chunks))],
    metadatas=[{"chunk": i} for i in range(len(chunks))]
)

# Invoke model
model_name = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=800)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# RAG Inference
def answer_question(question):
    # Retrieve chunks
    docs = retrieve_docs(question, k=3)
    context = "\n".join(docs)

    # Build prompt
    prompt = f"""
    Below are examples of warm, friendly explanations. Follow their style.

    Example 1:
    Q: What is abalone?
    A: Abalones are gentle sea creatures—beautiful marine snails with colorful shells. They live in cool ocean waters and play an important role in their ecosystems.

    Example 2:
    Q: Where do abalones live?
    A: Abalones can be found in ocean waters around the world, especially in cooler coastal areas. They're attached to rocks and enjoy peaceful, quiet habitats.

    Other Rules:
    - If the answer is not in the context, respond with: "I'm not seeing that information in the context provided."
    - NEVER guess.
    - NEVER add new facts.
    - NEVER rely on outside knowledge.
    - ONLY use the context, nothing else.
    Now answer the new question in the same friendly, warm, descriptive tone.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    return llm(prompt)

# Gradio UI
iface = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(lines=2, label="Ask a question about Abalones!"),
    outputs=gr.Textbox(label="Answer"),
    title="Abalone RAG QA Demo"
)

if __name__ == "__main__":
    iface.launch()
    
    # add a 'generating response...' to the UI