"""
RAG query system - retrieve relevant context from vector database
"""
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
from pathlib import Path

class RAGSystem:
    def __init__(self, db_path: str = 'vector_db.pkl'):
        """Initialize RAG system with vector database"""
        
        print(f"🔧 Loading vector database from {db_path}...")
        
        # Load vector DB
        with open(db_path, 'rb') as f:
            self.db = pickle.load(f)
        
        # Load embedding model (same one used to create DB)
        print(f"🔧 Loading embedding model: {self.db['model_name']}...")
        self.embedder = SentenceTransformer(self.db['model_name'])
        
        print(f"✅ RAG system ready: {self.db['stats']['total_docs']} documents loaded")
    
    def query(self, question: str, top_k: int = 5, doc_types: List[str] = None) -> List[Dict]:
        """
        Retrieve top-k relevant documents for a question
        
        Args:
            question: User's question
            top_k: Number of documents to return
            doc_types: Optional filter by document type 
                      (e.g., ['fire_event', 'major_fire'])
        
        Returns:
            List of relevant documents with similarity scores
        """
        
        # Embed the question
        q_embedding = self.embedder.encode([question])[0]
        
        # Get all embeddings from database
        db_embeddings = self.db['embeddings']
        
        # Calculate cosine similarity
        similarities = np.dot(db_embeddings, q_embedding) / (
            np.linalg.norm(db_embeddings, axis=1) * np.linalg.norm(q_embedding)
        )
        
        # Filter by document type if specified
        if doc_types:
            valid_indices = [
                i for i, doc in enumerate(self.db['documents'])
                if doc['type'] in doc_types
            ]
            
            if valid_indices:
                # Mask similarities for non-matching types
                mask = np.zeros(len(similarities), dtype=bool)
                mask[valid_indices] = True
                similarities = np.where(mask, similarities, -np.inf)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Build results
        results = []
        for idx in top_indices:
            if similarities[idx] == -np.inf:  # Skip filtered out docs
                continue
                
            doc = self.db['documents'][idx]
            results.append({
                'text': doc['text'],
                'score': float(similarities[idx]),
                'type': doc['type'],
                'metadata': doc.get('metadata', {})
            })
        
        return results
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        return self.db['stats']
    
    def search_by_date(self, date_str: str, top_k: int = 5) -> List[Dict]:
        """Search for events on a specific date"""
        
        results = []
        for i, doc in enumerate(self.db['documents']):
            if doc['type'] == 'fire_event' and doc.get('date') == date_str:
                results.append({
                    'text': doc['text'],
                    'score': 1.0,
                    'type': doc['type'],
                    'metadata': doc['metadata']
                })
        
        return results[:top_k]
    
    def search_by_location(self, lat: float, lon: float, radius_km: float = 50, top_k: int = 10) -> List[Dict]:
        """Search for events near a location"""
        
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            return R * 2 * np.arcsin(np.sqrt(a))
        
        results = []
        for doc in self.db['documents']:
            metadata = doc.get('metadata', {})
            doc_lat = metadata.get('lat')
            doc_lon = metadata.get('lon')
            
            if doc_lat and doc_lon:
                distance = haversine(lat, lon, doc_lat, doc_lon)
                if distance <= radius_km:
                    results.append({
                        'text': doc['text'],
                        'score': 1.0 - (distance / radius_km),  # Closer = higher score
                        'type': doc['type'],
                        'metadata': metadata,
                        'distance_km': round(distance, 2)
                    })
        
        # Sort by distance
        results.sort(key=lambda x: x['distance_km'])
        return results[:top_k]


# Test function
def test_rag():
    """Test the RAG system"""
    
    print("\n" + "="*60)
    print("Testing RAG System")
    print("="*60 + "\n")
    
    rag = RAGSystem()
    
    # Test queries
    test_queries = [
        "What happened during the Camp Fire?",
        "How many hospitals are in California?",
        "What are high wind conditions?",
        "Tell me about fires in 2023",
        "What's the risk when fires are close?"
    ]
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 60)
        
        results = rag.query(query, top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [Score: {result['score']:.3f}] [{result['type']}]")
            print(f"   {result['text'][:200]}...")
            if result['metadata']:
                print(f"   Metadata: {result['metadata']}")
    
    print("\n" + "="*60)
    print("✅ RAG Test Complete")
    print("="*60)


if __name__ == "__main__":
    test_rag()