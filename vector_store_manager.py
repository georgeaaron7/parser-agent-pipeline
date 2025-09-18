from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import sqlite3
import json
import uuid
from typing import List, Dict, Any, Optional
import numpy as np
import gc
import torch
from database_manager import DatabaseManager

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import pickle
import os

class VectorStoreManager:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333, db_path="rag_database.db", 
                 use_gpu=False, batch_size=16, embedding_model="lightweight"):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.db_path = db_path
        self.collection_name = "pdf_content"
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.batch_size = batch_size
        self.device = 'cuda' if self.use_gpu else 'cpu'
        self.db_manager = DatabaseManager(db_path)
        
        # Initialize embedding model and get existing collection info BEFORE init
        existing_dim = self._get_existing_collection_dimension()
        self._init_embedding_model(embedding_model, existing_dim)
        self.init_vector_collection()
    
    def _get_existing_collection_dimension(self) -> Optional[int]:
        """Get dimension of existing collection to maintain consistency"""
        try:
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)
            
            if collection_exists:
                existing = self.client.get_collection(self.collection_name)
                existing_dim = existing.config.params.vectors.size
                existing_count = existing.points_count
                print(f"Found existing collection: {existing_count} vectors, dim={existing_dim}")
                return existing_dim
        except Exception as e:
            print(f"Error checking existing collection: {e}")
        
        return None
    
    def _init_embedding_model(self, model_type="lightweight", existing_dim=None):
        """Initialize embedding model, adapting to existing collection if needed"""
        
        # If we have existing vectors, prefer SentenceTransformers for consistency
        if existing_dim == 384:
            print(f"Adapting to existing collection dimension: {existing_dim}")
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._init_cpu_sentence_transformer()
                return
            else:
                print("SentenceTransformers not available, cannot use existing 384-dim collection")
        
        # Otherwise initialize based on requested type
        if model_type == "lightweight" and not existing_dim:
            if SKLEARN_AVAILABLE:
                self._init_tfidf_embedding()
            else:
                print("scikit-learn not available, falling back to SentenceTransformers")
                self._init_cpu_sentence_transformer()
        elif model_type == "cpu_optimized" or existing_dim == 384:
            self._init_cpu_sentence_transformer()
        elif model_type == "gpu_optimized" and self.use_gpu:
            self._init_gpu_sentence_transformer()
        else:
            # Default fallback
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._init_cpu_sentence_transformer()
            else:
                self._init_tfidf_embedding()
    
    def _init_tfidf_embedding(self):
        """Initialize TF-IDF with adaptive dimensions"""
        self.embedding_method = "tfidf"
        self.model_path = "lightweight_tfidf_model.pkl"
        
        # Try to load existing model first
        if self._try_load_existing_tfidf_model():
            return
        
        # Initialize with reasonable defaults - will be adapted when fitting
        self.embedding_dim = 128  # Start with smaller dimension
        self.vectorizer = TfidfVectorizer(
            max_features=200,  # Reduced for small datasets
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        self.svd = None  # Will be initialized with correct dimensions
        self.is_fitted = False
        
        print(f"Initialized TF-IDF embeddings (adaptive dimensions)")
    
    def _try_load_existing_tfidf_model(self) -> bool:
        """Try to load existing TF-IDF model"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.svd = model_data['svd']
                    self.embedding_dim = model_data.get('n_components', self.svd.n_components)
                    self.is_fitted = True
                    print(f"Loaded existing TF-IDF model (dim: {self.embedding_dim})")
                    return True
            except Exception as e:
                print(f"Failed to load TF-IDF model: {e}")
        return False
    
    def _init_cpu_sentence_transformer(self):
        """CPU-optimized sentence transformer"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.model.max_seq_length = 256
            self.embedding_dim = 384
            self.embedding_method = "sentence_transformers_cpu"
            print("Using CPU-optimized SentenceTransformers (384 dim)")
        except Exception as e:
            print(f"Failed to load CPU SentenceTransformers: {e}")
            if SKLEARN_AVAILABLE:
                self._init_tfidf_embedding()
            else:
                raise Exception("No embedding models available")
    
    def _init_gpu_sentence_transformer(self):
        """GPU-optimized sentence transformer"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
            self.model.max_seq_length = 128
            self.embedding_dim = 384
            self.embedding_method = "sentence_transformers_gpu"
            if hasattr(torch, 'cuda') and torch.cuda.is_available():
                try:
                    self.model.half()
                    print("Using GPU-optimized SentenceTransformers with FP16 (384 dim)")
                except:
                    print("Using GPU-optimized SentenceTransformers with FP32 (384 dim)")
        except Exception as e:
            print(f"Failed to load GPU SentenceTransformers: {e}")
            self._init_cpu_sentence_transformer()
    
    def _clear_gpu_cache(self):
        """Clear GPU cache to prevent memory issues"""
        if self.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    def _fit_tfidf_model_from_texts(self, texts: List[str]) -> bool:
        """Fit TF-IDF model with adaptive dimensions based on actual data"""
        if not texts:
            print("No texts provided for TF-IDF fitting")
            return False
        
        try:
            print(f"Fitting TF-IDF model with {len(texts)} text samples...")
            
            # Fit vectorizer
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            n_features = tfidf_matrix.shape[1]
            
            print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
            print(f"Available features: {n_features}")
            
            # Adaptive SVD components - must be less than n_features
            n_components = min(128, n_features - 1, len(texts) - 1)
            n_components = max(n_components, 10)  # Minimum 10 components
            
            print(f"Using {n_components} SVD components")
            
            # Initialize and fit SVD
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd.fit(tfidf_matrix)
            
            # Update embedding dimension
            self.embedding_dim = n_components
            self.is_fitted = True
            
            # Save model
            try:
                with open(self.model_path, 'wb') as f:
                    pickle.dump({
                        'vectorizer': self.vectorizer,
                        'svd': self.svd,
                        'n_components': n_components
                    }, f)
                print(f"Saved TF-IDF model (dim: {self.embedding_dim})")
            except Exception as e:
                print(f"Warning: Could not save TF-IDF model: {e}")
            
            # Test the model
            test_embedding = self._encode_tfidf(texts[0])
            non_zero_count = sum(1 for x in test_embedding if abs(x) > 1e-10)
            print(f"Model test: {non_zero_count}/{len(test_embedding)} non-zero values")
            
            return True
            
        except Exception as e:
            print(f"Error fitting TF-IDF model: {e}")
            self.is_fitted = False
            return False
    
    def _get_texts_from_database(self) -> List[str]:
        """Get text samples from database for model fitting"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT text_content FROM text_content 
            WHERE text_content IS NOT NULL AND LENGTH(text_content) > 10
            ORDER BY id
            ''')
            
            texts = [row[0] for row in cursor.fetchall()]
            conn.close()
            return texts
        except Exception as e:
            print(f"Error getting texts from database: {e}")
            return []
    
    def encode(self, text: str) -> List[float]:
        """Text encoding with improved error handling"""
        if self.embedding_method == "tfidf":
            return self._encode_tfidf(text)
        elif "sentence_transformers" in self.embedding_method:
            return self._encode_sentence_transformer(text)
        else:
            print("Warning: No valid embedding method")
            return [0.0] * self.embedding_dim
    
    def _encode_tfidf(self, text: str) -> List[float]:
        """TF-IDF encoding with automatic fitting"""
        if not self.is_fitted:
            print("TF-IDF model not fitted, attempting to fit from database...")
            texts = self._get_texts_from_database()
            if texts:
                success = self._fit_tfidf_model_from_texts(texts)
                if not success:
                    print("Failed to fit TF-IDF model")
                    return [0.0] * self.embedding_dim
            else:
                print("No text data available for fitting")
                return [0.0] * self.embedding_dim
        
        try:
            if not text or not text.strip():
                return [0.0] * self.embedding_dim
            
            tfidf_vec = self.vectorizer.transform([text])
            embedding = self.svd.transform(tfidf_vec)
            return embedding[0].tolist()
        except Exception as e:
            print(f"Error in TF-IDF encoding: {e}")
            return [0.0] * self.embedding_dim
    
    def _encode_sentence_transformer(self, text: str) -> List[float]:
        """Sentence transformer encoding"""
        try:
            if len(text) > 500:
                text = text[:500]
            with torch.no_grad():
                if self.use_gpu:
                    embedding = self.model.encode(text, show_progress_bar=False, 
                                                convert_to_numpy=True, normalize_embeddings=True)
                else:
                    embedding = self.model.encode(text, show_progress_bar=False, 
                                                convert_to_tensor=False)
            
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
            
        except Exception as e:
            print(f"Error encoding text: {e}")
            return [0.0] * self.embedding_dim
        finally:
            self._clear_gpu_cache()
    
    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch encoding"""
        if self.embedding_method == "tfidf":
            return [self._encode_tfidf(text) for text in texts]
        
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            batch_texts = [text[:500] for text in batch_texts]
            try:
                with torch.no_grad():
                    if self.use_gpu:
                        batch_embeddings = self.model.encode(batch_texts, 
                                                           show_progress_bar=False,
                                                           convert_to_numpy=True,
                                                           normalize_embeddings=True)
                    else:
                        batch_embeddings = self.model.encode(batch_texts,
                                                           show_progress_bar=False,
                                                           convert_to_tensor=False)
                
                embeddings.extend(batch_embeddings.tolist() if hasattr(batch_embeddings, 'tolist') 
                                else batch_embeddings)
                
            except Exception as e:
                print(f"Error in batch encoding: {e}")
                for text in batch_texts:
                    embeddings.append(self.encode(text))
            finally:
                self._clear_gpu_cache()
        return embeddings
    
    def init_vector_collection(self):
        """Initialize Qdrant collection - SAFE version that preserves existing data"""
        try:
            collections = self.client.get_collections()
            collection_exists = any(c.name == self.collection_name for c in collections.collections)

            if not collection_exists:
                # Create new collection
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                )
                print(f"Created Qdrant collection: {self.collection_name} (dim={self.embedding_dim})")
            else:
                existing = self.client.get_collection(self.collection_name)
                existing_dim = existing.config.params.vectors.size
                existing_count = existing.points_count
                
                if existing_dim != self.embedding_dim:
                    print(f"Dimension mismatch: Qdrant={existing_dim}, model={self.embedding_dim}")
                    
                    if existing_count == 0:
                        # Safe to recreate empty collection
                        print("Collection is empty, recreating with new dimensions...")
                        self.client.delete_collection(self.collection_name)
                        self.client.create_collection(
                            collection_name=self.collection_name,
                            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.COSINE)
                        )
                        print(f"Recreated collection with dim={self.embedding_dim}")
                    else:
                        # Adapt model to existing collection to preserve data
                        print(f"Collection has {existing_count} vectors, adapting model to existing dim={existing_dim}")
                        self._adapt_model_to_dimension(existing_dim)
                else:
                    print(f"Collection exists with correct dim={existing_dim}, {existing_count} vectors")

        except Exception as e:
            print(f"Error initializing Qdrant collection: {str(e)}")
    
    def _adapt_model_to_dimension(self, target_dim: int):
        """Adapt embedding model to match existing collection dimension"""
        if target_dim == 384 and SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Switching to SentenceTransformers to match existing 384-dim collection")
            self._init_cpu_sentence_transformer()
        elif self.embedding_method == "tfidf":
            print(f"Adapting TF-IDF model to dimension {target_dim}")
            self.embedding_dim = target_dim
            # Reinitialize SVD with target dimension if model is fitted
            if hasattr(self, 'is_fitted') and self.is_fitted:
                texts = self._get_texts_from_database()
                if texts:
                    self._fit_tfidf_model_with_target_dim(texts, target_dim)
        else:
            print(f"Cannot adapt {self.embedding_method} to dimension {target_dim}")
    
    def _fit_tfidf_model_with_target_dim(self, texts: List[str], target_dim: int):
        """Fit TF-IDF model with specific target dimension"""
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            n_features = tfidf_matrix.shape[1]
            
            # Use target dimension but ensure it's valid
            n_components = min(target_dim, n_features - 1, len(texts) - 1)
            n_components = max(n_components, 10)
            
            self.svd = TruncatedSVD(n_components=n_components, random_state=42)
            self.svd.fit(tfidf_matrix)
            self.embedding_dim = n_components
            self.is_fitted = True
            
            print(f"Fitted TF-IDF model with target dimension {n_components}")
        except Exception as e:
            print(f"Error fitting TF-IDF with target dimension: {e}")
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 40) -> List[str]:
        """Text chunking"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                sentence_end = text.rfind('. ', start, end)
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    word_end = text.rfind(' ', start, end)
                    if word_end > start:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
            
        return chunks
    
    def clear_document_vectors(self, document_id: int):
        """Clear all vectors for a specific document"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT vector_id FROM text_content WHERE document_id = ?', (document_id,))
            vector_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if vector_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=vector_ids
                )
                print(f"Deleted {len(vector_ids)} vectors for document {document_id}")
        except Exception as e:
            print(f"Error clearing document vectors: {e}")
    
    def store_text_content(self, document_id: int, text_pages: List[Dict]):
        """Store text content with robust error handling"""
        points = []
        text_records = []
        all_texts = []
        
        print(f"Preparing text content for document {document_id}...")
        
        # Collect all texts for model fitting if needed
        for page_info in text_pages:
            text_content = page_info['text']
            if text_content.strip():
                chunks = self.chunk_text(text_content)
                all_texts.extend(chunks)
        
        print(f"Processing {len(all_texts)} text chunks...")
        
        # Ensure model is ready
        if self.embedding_method == "tfidf" and not self.is_fitted:
            print("Fitting TF-IDF model...")
            success = self._fit_tfidf_model_from_texts(all_texts)
            if not success:
                print("Failed to fit TF-IDF model, cannot store text content")
                return
        
        # Process pages
        batch_texts = []
        batch_metadata = []
        
        for page_info in text_pages:
            page_number = page_info['page']
            image_path = page_info['path']
            text_content = page_info['text']
            
            if not text_content.strip():
                continue
            
            chunks = self.chunk_text(text_content)
            
            for i, chunk in enumerate(chunks):
                batch_texts.append(chunk)
                batch_metadata.append({
                    'document_id': document_id,
                    'page_number': page_number,
                    'source_file': image_path.split('/')[-1],
                    'chunk_index': i,
                    'text_content': chunk
                })
                
                if len(batch_texts) >= self.batch_size:
                    self._process_batch(batch_texts, batch_metadata, points, text_records)
                    batch_texts = []
                    batch_metadata = []
        
        # Process remaining batch
        if batch_texts:
            self._process_batch(batch_texts, batch_metadata, points, text_records)
        
        # Store in databases
        self._store_in_databases(points, text_records)
        self._clear_gpu_cache()
    
    def _process_batch(self, texts, metadata, points, text_records):
        """Process a batch of texts"""
        try:
            if self.embedding_method == "tfidf":
                embeddings = [self.encode(text) for text in texts]
            else:
                embeddings = self.encode_batch(texts)
            
            for text, meta, embedding in zip(texts, metadata, embeddings):
                # Skip zero vectors
                if all(abs(x) < 1e-10 for x in embedding):
                    print(f"Skipping zero vector for text: {text[:50]}...")
                    continue
                
                vector_id = str(uuid.uuid4())
                point = PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload={
                        'document_id': meta['document_id'],
                        'page_number': meta['page_number'],
                        'source_file': meta['source_file'],
                        'text_content': text,
                        'text_type': 'body_text',
                        'chunk_index': meta['chunk_index']
                    }
                )
                points.append(point)
                
                text_records.append((
                    vector_id, meta['document_id'], meta['page_number'], 
                    meta['source_file'], text, 'body_text', meta['chunk_index']
                ))
        except Exception as e:
            print(f"Error processing batch: {e}")
    
    def _store_in_databases(self, points, text_records):
        """Store processed data in databases"""
        if points:
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                print(f"Stored {len(points)} text chunks in Qdrant")
            except Exception as e:
                print(f"Error storing in Qdrant: {str(e)}")
        
        if text_records:
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.executemany('''
                INSERT INTO text_content (vector_id, document_id, page_number, source_file, text_content, text_type, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', text_records)
                
                conn.commit()
                conn.close()
                print(f"Stored {len(text_records)} text records in SQLite")
            except Exception as e:
                print(f"Error storing in SQLite: {str(e)}")
    
    def search_similar_text(self, query: str, limit: int = 5, document_id: Optional[int] = None) -> List[Dict]:
        """Similarity search with improved error handling"""
        try:
            query_embedding = self.encode(query)
            
            if all(abs(x) < 1e-10 for x in query_embedding):
                print("Query embedding is zero vector, using fallback search")
                return self._fallback_text_search(query, limit, document_id)
            
            search_filter = None
            if document_id is not None:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                search_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id)
                        )
                    ]
                )
            
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit,
                query_filter=search_filter,
                with_payload=True
            )
            
            results = []
            for result in search_results:
                results.append({
                    'id': result.id,
                    'score': result.score,
                    'document_id': result.payload['document_id'],
                    'page_number': result.payload['page_number'],
                    'source_file': result.payload['source_file'],
                    'text_content': result.payload['text_content'],
                    'text_type': result.payload['text_type'],
                    'chunk_index': result.payload['chunk_index']
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching vectors: {str(e)}")
            return self._fallback_text_search(query, limit, document_id)
        finally:
            self._clear_gpu_cache()
    
    def _fallback_text_search(self, query: str, limit: int, document_id: Optional[int] = None) -> List[Dict]:
        """Fallback text search using SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if document_id is not None:
                cursor.execute('''
                SELECT vector_id, document_id, page_number, source_file, text_content, text_type, chunk_index
                FROM text_content
                WHERE text_content LIKE ? AND document_id = ?
                ORDER BY page_number
                LIMIT ?
                ''', (f'%{query}%', document_id, limit))
            else:
                cursor.execute('''
                SELECT vector_id, document_id, page_number, source_file, text_content, text_type, chunk_index
                FROM text_content
                WHERE text_content LIKE ?
                ORDER BY page_number
                LIMIT ?
                ''', (f'%{query}%', limit))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'id': row[0],
                    'score': 0.5,
                    'document_id': row[1],
                    'page_number': row[2],
                    'source_file': row[3],
                    'text_content': row[4],
                    'text_type': row[5],
                    'chunk_index': row[6]
                })
            
            conn.close()
            return results
            
        except Exception as e:
            print(f"Error in fallback search: {str(e)}")
            return []
    
    def cleanup(self):
        """Clean up resources"""
        self._clear_gpu_cache()
        if hasattr(self, 'model'):
            del self.model
        gc.collect()
