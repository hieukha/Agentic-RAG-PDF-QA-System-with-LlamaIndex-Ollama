#!/usr/bin/env python3
"""
Setup script for Qdrant vector database integration
"""

import subprocess
import sys
import time
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def check_qdrant_connection():
    """Check if Qdrant is running and accessible"""
    try:
        response = requests.get("http://localhost:6333/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def setup_qdrant_collection():
    """Setup Qdrant collection for the application"""
    try:
        client = QdrantClient(host="localhost", port=6333)
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        collection_name = "pdf_qa_collection"
        vector_size = 768  # nomic-embed-text dimension
        
        if collection_name not in collection_names:
            # Create collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )
            print(f"✅ Created Qdrant collection: {collection_name}")
        else:
            print(f"✅ Using existing Qdrant collection: {collection_name}")
            
        return True
    except Exception as e:
        print(f"❌ Error setting up Qdrant collection: {e}")
        return False

def main():
    print("🚀 Setting up Qdrant vector database...")
    
    # Check if Qdrant is running
    if not check_qdrant_connection():
        print("❌ Qdrant is not running. Please start it first:")
        print("   docker-compose up -d qdrant")
        print("   or")
        print("   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant")
        sys.exit(1)
    
    # Setup collection
    if setup_qdrant_collection():
        print("✅ Qdrant setup completed successfully!")
        print("\n📝 Next steps:")
        print("1. Make sure Ollama is running with phi3 and nomic-embed-text models")
        print("2. Run: python app.py")
    else:
        print("❌ Failed to setup Qdrant collection")
        sys.exit(1)

if __name__ == "__main__":
    main()
