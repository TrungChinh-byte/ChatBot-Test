import os
from uuid import uuid4
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_core.documents import Document # Quan trọng để tạo object Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from collections import defaultdict

# Load envs
load_dotenv() 

# chroma director
CHROMA_PATH = r"chroma_db"

# SSMS load
DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
# SQL - View
TABLE_OR_VIEW_NAME = "[dbo].[view_post_info]" 

CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "car_vectorize"

BATCH_SIZE = 1000   # an toàn cho Chroma
CHUNK_SIZE = 250
CHUNK_OVERLAP = 30

def from_sql_to_json():
    print(f"🔄 Đang kết nối đến SQL Server...")
    engine = create_engine(DB_CONNECTION_STRING)
    try:
        with engine.connect() as con:
            result = con.execute(text(f"SELECT * FROM {TABLE_OR_VIEW_NAME}"))
            rows = result.mappings().all()
            
            first_row = rows[0]
            all_cols = list(first_row.keys())

            key_list = all_cols[:-2]
            
            cars = {}  

            for row in rows:
                VIN = row["VIN"]
                if VIN not in cars:
                    base_info = {col: row[col] for col in key_list}
                    base_info["features"] = defaultdict(list)
                    cars[VIN] = base_info
                    
                car = cars[VIN]

                feature_type = row["feature_type"]   
                feature_name = row["feature_name"]  

                if feature_name and feature_name not in car["features"][feature_type]:
                    car["features"][feature_type].append(feature_name)

            output = []
            for car in cars.values():
                car["features"] = dict(car["features"])
                output.append(car)

            return output
    except Exception as e:
        print(f"❌ Lỗi kết nối SQL: {e}")
        return []
        


# Load database to document
def load_documents_from_sql():
    documents = []
    rows = from_sql_to_json()
    for row_dict in rows:
        # Select column to vectorize
        page_content = f"""status: {row_dict.get('status', '')}. title: {row_dict.get('title', '')}
        .brand: {row_dict.get('brand', '')}. interior_color: {row_dict.get('interior_color', '')}
        . exterior_color: {row_dict.get('exterior_color', '')}. drivetrain: {row_dict.get('drivetrain', '')} . fuel_type: {row_dict.get('fuel_type', '')}.
        transmission: {row_dict.get('transmission', '')}.  engine: {row_dict.get('engine', '')}. features : {row_dict.get('features', '')}
        """
        
        # 2. Tạo metadata (Dùng để lọc: Giá, Năm, Người bán...)
        metadata = {
            "VIN" : row_dict.get('VIN'),
            "Price": row_dict.get('price'),
            "Monthly Payment": row_dict.get('monthly_payment'),
            "Mileage" : row_dict.get('mileage'),
            "Miles Per Gallon" : row_dict.get('mpg'),
            "Post Link" : row_dict.get('post_link'),
            "Status" : row_dict.get('status', ''),
            "Title" : row_dict.get('title', ''),
            "Brand" : row_dict.get('brand', ''),
            "Interior Color": row_dict.get('interior_color', ''),
            "Exterior Color" : row_dict.get('exterior_color', ''),
            "Drivetrain": row_dict.get('drivetrain', ''),
            "Fuel Type" : row_dict.get('fuel_type', ''),
            "Transmission" : row_dict.get('transmission', ''),
            "Engine": row_dict.get('engine', '')
        }

        # Tạo Document object của LangChain
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)
                
    print(f"✅ Đã tải xong {len(documents)} dòng từ Database.")
    return documents


# =====================================================
# CHROMA BATCH ADD
# =====================================================
def add_documents_in_batches(vector_store, documents, batch_size=BATCH_SIZE):
    total = len(documents)
    for i in range(0, total, batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = [str(uuid4()) for _ in batch_docs]

        vector_store.add_documents(
            documents=batch_docs,
            ids=batch_ids
        )

        print(f"💾 Added {min(i + batch_size, total)}/{total}")

# =====================================================
# MAIN
# =====================================================
def main():
    print("🚀 Starting SQL → Chroma ingest...")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH,
    )

    documents = load_documents_from_sql()

    if not documents:
        print("⚠️ No data found. Exit.")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)
    print(f"✂️ Split into {len(chunks)} chunks")

    add_documents_in_batches(vector_store, chunks)

    print("✅ Ingest completed successfully!")

if __name__ == "__main__":
    main()