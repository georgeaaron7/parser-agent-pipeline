import sqlite3
import hashlib
import json
import os
from typing import Dict, List, Any, Optional

class DatabaseManager:
    def __init__(self, db_path="rag_database.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database with all required tables including document tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT UNIQUE,
            file_path TEXT,
            file_hash TEXT UNIQUE,
            file_size INTEGER,
            page_count INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_status TEXT DEFAULT 'completed'
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            page_number INTEGER,
            source_file TEXT,
            table_data TEXT,
            description TEXT,
            columns TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS formulas (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            page_number INTEGER,
            source_file TEXT,
            original_formula TEXT,
            parsed_formula TEXT,
            formula_type TEXT,
            variables TEXT,
            description TEXT,
            executable_code TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
        ''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS text_content (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id INTEGER,
            vector_id TEXT UNIQUE,
            page_number INTEGER,
            source_file TEXT,
            text_content TEXT,
            text_type TEXT,
            chunk_index INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents (id)
        )
        ''')
        
        conn.commit()
        conn.close()
        print("Database initialized with document tracking")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get file information including size and hash"""
        try:
            stat = os.stat(file_path)
            return {
                'size': stat.st_size,
                'hash': self.calculate_file_hash(file_path),
                'filename': os.path.basename(file_path),
                'path': file_path
            }
        except Exception as e:
            print(f"Error getting file info for {file_path}: {e}")
            return {'size': 0, 'hash': '', 'filename': '', 'path': file_path}
    
    def check_document_exists(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Check if document with same content already exists in database"""
        file_info = self.get_file_info(file_path)
        if not file_info['hash']:
            return None
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
        SELECT id, filename, file_path, file_hash, file_size, page_count, processed_at, processing_status
        FROM documents 
        WHERE file_hash = ?
        ''', (file_info['hash'],))
        
        result = cursor.fetchone()
        
        if result:
            conn.close()
            return {
                'id': result[0],
                'filename': result[1],
                'file_path': result[2],
                'file_hash': result[3],
                'file_size': result[4],
                'page_count': result[5],
                'processed_at': result[6],
                'processing_status': result[7],
                'exists': True,
                'same_content': True
            }
        
        cursor.execute('''
        SELECT id, filename, file_path, file_hash, file_size, page_count, processed_at, processing_status
        FROM documents 
        WHERE filename = ? AND file_size = ?
        ''', (file_info['filename'], file_info['size']))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'filename': result[1],
                'file_path': result[2],
                'file_hash': result[3],
                'file_size': result[4],
                'page_count': result[5],
                'processed_at': result[6],
                'processing_status': result[7],
                'exists': True,
                'same_content': False 
            }
        
        return None
    
    def register_document(self, file_path: str, page_count: int) -> int:
        """Register a new document in the database"""
        file_info = self.get_file_info(file_path)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT INTO documents (filename, file_path, file_hash, file_size, page_count)
            VALUES (?, ?, ?, ?, ?)
            ''', (file_info['filename'], file_info['path'], file_info['hash'], 
                  file_info['size'], page_count))
            
            document_id = cursor.lastrowid
            conn.commit()
            print(f"Registered document: {file_info['filename']} (ID: {document_id})")
            return document_id
            
        except sqlite3.IntegrityError:
            cursor.execute('SELECT id FROM documents WHERE file_hash = ?', (file_info['hash'],))
            result = cursor.fetchone()
            if result:
                print(f"Document already registered: {file_info['filename']} (ID: {result[0]})")
                return result[0]
            raise
        finally:
            conn.close()
    
    def get_document_data_counts(self, document_id: int) -> Dict[str, int]:
        """Get counts of existing data for a document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        counts = {}
        cursor.execute('SELECT COUNT(*) FROM tables WHERE document_id = ?', (document_id,))
        counts['tables'] = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM formulas WHERE document_id = ?', (document_id,))
        counts['formulas'] = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM text_content WHERE document_id = ?', (document_id,))
        counts['text_chunks'] = cursor.fetchone()[0]
        
        conn.close()
        return counts
    
    def clear_document_data(self, document_id: int):
        """Clear all data for a specific document"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM text_content WHERE document_id = ?', (document_id,))
        cursor.execute('DELETE FROM formulas WHERE document_id = ?', (document_id,))
        cursor.execute('DELETE FROM tables WHERE document_id = ?', (document_id,))
        
        conn.commit()
        conn.close()
        print(f"Cleared existing data for document ID: {document_id}")
    
    def update_document_status(self, document_id: int, status: str):
        """Update document processing status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        UPDATE documents 
        SET processing_status = ?, processed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        ''', (status, document_id))
        
        conn.commit()
        conn.close()
