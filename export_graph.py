import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from generate_response import _build_agentic_app, CHROMA_PATH, COLLECTION_NAME

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=CHROMA_PATH,
)

app = _build_agentic_app(llm, vector_store)
png_data = app.get_graph().draw_mermaid_png()

output_path = "graph.png"
with open(output_path, "wb") as f:
    f.write(png_data)

print(f"Graph exported to: {os.path.abspath(output_path)}")
