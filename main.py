import os
import sys
import argparse
import json
import gc
import sqlite3
from pdf_converter import PDFToImageConverter
from content_detector import ContentDetector
from table_extractor import TableExtractor
from formula_processor import FormulaProcessor
from database_manager import DatabaseManager

try:
    from vector_store_manager import VectorStoreManager
    VECTOR_STORE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Updated vector store not available: {e}")
    try:
        from vector_store_manager import VectorStoreManager
        VECTOR_STORE_AVAILABLE = True
        print("Using original vector store manager")
    except ImportError:
        VECTOR_STORE_AVAILABLE = False
        print("No vector store available")

from rag_pipeline import RAGPipeline

class SmartPDFRAGOrchestrator:
    def __init__(self, output_dir="pdf_processing", db_path="rag_database.db", 
                 gpu_mode="auto", batch_size=4, ocr_mode="lightweight", force_reprocess=False):
        self.output_dir = output_dir
        self.db_path = db_path
        self.image_dir = os.path.join(output_dir, "images")
        self.analysis_file = os.path.join(output_dir, "content_analysis.json")
        self.force_reprocess = force_reprocess
        self.gpu_mode = gpu_mode
        self.batch_size = batch_size
        self.ocr_mode = ocr_mode
        self.use_gpu = self._determine_gpu_usage()
        self.db_manager = DatabaseManager(db_path)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        
        print(f"Smart PDF RAG Orchestrator initialized:")
        print(f"  GPU Mode: {self.gpu_mode}")
        print(f"  Using GPU: {self.use_gpu}")
        print(f"  OCR Mode: {self.ocr_mode}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Force Reprocess: {self.force_reprocess}")
    
    def _determine_gpu_usage(self):
        """Determine if GPU should be used based on availability and mode"""
        if self.gpu_mode == "disable":
            return False
        
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            
            if self.gpu_mode == "auto":
                if gpu_available:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    print(f"GPU detected: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
                    if gpu_memory >= 4.0:  
                        return True
                    else:
                        print("GPU memory insufficient for optimization, using CPU")
                        return False
                else:
                    print("No GPU detected, using CPU")
                    return False
            
            elif self.gpu_mode == "force":
                if gpu_available:
                    return True
                else:
                    print("GPU forced but not available, falling back to CPU")
                    return False
            
        except ImportError:
            print("PyTorch not available, using CPU")
            return False
        
        return False
    
    def _clear_memory(self):
        """Clear memory and GPU cache"""
        gc.collect()
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
    
    def check_and_handle_existing_document(self, pdf_path: str) -> tuple:
        """
        Check if document exists and determine processing strategy
        Returns: (document_id, should_process, reason)
        """
        print("\n" + "="*50)
        print("CHECKING DOCUMENT STATUS")
        print("="*50)
        
        existing_doc = self.db_manager.check_document_exists(pdf_path)
        
        if existing_doc is None:
            print("✓ New document - will process from scratch")
            return None, True, "new_document"
        
        elif existing_doc['same_content'] and not self.force_reprocess:
            data_counts = self.db_manager.get_document_data_counts(existing_doc['id'])
            print(f"✓ Document already processed with same content:")
            print(f"  Document ID: {existing_doc['id']}")
            print(f"  Filename: {existing_doc['filename']}")
            print(f"  Processed: {existing_doc['processed_at']}")
            print(f"  Tables: {data_counts['tables']}")
            print(f"  Formulas: {data_counts['formulas']}")
            print(f"  Text chunks: {data_counts['text_chunks']}")
            print("→ Using existing data (skip with --force to reprocess)")
            return existing_doc['id'], False, "same_content"
        
        elif existing_doc['same_content'] and self.force_reprocess:
            print(f"⚠ Document exists with same content but force reprocess enabled")
            print(f"  Document ID: {existing_doc['id']}")
            print("→ Clearing existing data and reprocessing")
            self._clear_existing_data(existing_doc['id'])
            return existing_doc['id'], True, "forced_reprocess"
        
        else:
            print(f"⚠ Document with same name but different content found:")
            print(f"  Existing ID: {existing_doc['id']}")
            print(f"  Existing hash: {existing_doc['file_hash'][:16]}...")
            print("→ Clearing old data and processing new content")
            self._clear_existing_data(existing_doc['id'])
            return existing_doc['id'], True, "different_content"
    
    def _clear_existing_data(self, document_id: int):
        """Clear existing data for a document"""
        print(f"Clearing existing data for document {document_id}...")
        if VECTOR_STORE_AVAILABLE:
            try:
                vector_manager = VectorStoreManager(
                    db_path=self.db_path,
                    use_gpu=False,  
                    embedding_model="lightweight"
                )
                vector_manager.clear_document_vectors(document_id)
                vector_manager.cleanup()
            except Exception as e:
                print(f"Warning: Could not clear vector data: {e}")
        self.db_manager.clear_document_data(document_id)
        self._clear_memory()
    
    def process_pdf(self, pdf_path: str, dpi: int = 300):
        """
        Smart PDF processing pipeline with document existence checking
        """
        print("="*60)
        print("SMART PDF RAG PROCESSING PIPELINE")
        print("="*60)
        
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found: {pdf_path}")
            return False
        
        try:
            document_id, should_process, reason = self.check_and_handle_existing_document(pdf_path)
            
            if not should_process:
                print("\n" + "="*60)
                print("PROCESSING SKIPPED - USING EXISTING DATA")
                print("="*60)
                return True
            print(f"\n1. Converting PDF to images... (Reason: {reason})")
            converter = PDFToImageConverter(self.image_dir)
            image_paths = converter.convert_pdf_to_images(pdf_path, dpi)
            
            if not image_paths:
                print("Error: No images were created from PDF")
                return False
            
            print(f"✓ Created {len(image_paths)} page images")
            if document_id is None:
                document_id = self.db_manager.register_document(pdf_path, len(image_paths))
                print(f"✓ Registered new document (ID: {document_id})")
            else:
                self.db_manager.update_document_status(document_id, "processing")
                print(f"✓ Updated document status (ID: {document_id})")
            
            self._clear_memory()
            print(f"\n2. Analyzing content types using {self.ocr_mode} mode...")
            detector = ContentDetector(
                ocr_mode=self.ocr_mode,
                use_gpu=self.use_gpu,
                batch_size=self.batch_size
            )
            
            try:
                analysis_results = detector.analyze_images(image_paths)
                with open(self.analysis_file, 'w') as f:
                    json.dump(analysis_results, f, indent=2)
                
                print(f"✓ Analysis complete:")
                print(f"  - Pages with tables: {len(analysis_results['table_pages'])}")
                print(f"  - Pages with formulas: {len(analysis_results['formula_pages'])}")
                print(f"  - Pages with text: {len(analysis_results['text_pages'])}")
                
            finally:
                detector.cleanup()
                self._clear_memory()
            if analysis_results['table_pages']:
                print(f"\n3. Extracting tables from {len(analysis_results['table_pages'])} pages...")
                table_extractor = TableExtractor(self.db_path)
                table_extractor.process_table_pages(document_id, analysis_results['table_pages'])
                print(f"✓ Processed tables")
                self._clear_memory()
            else:
                print("\n3. No tables found to extract")
            if analysis_results['formula_pages']:
                print(f"\n4. Processing formulas from {len(analysis_results['formula_pages'])} pages...")
                formula_processor = FormulaProcessor(self.db_path)
                formula_processor.process_formula_pages(document_id, analysis_results['formula_pages'])
                print(f"✓ Processed formulas")
                self._clear_memory()
            else:
                print("\n4. No formulas found to process")
            if analysis_results['text_pages'] and VECTOR_STORE_AVAILABLE:
                print(f"\n5. Storing text in vector database...")
                if self.use_gpu:
                    vector_manager = VectorStoreManager(
                        db_path=self.db_path,
                        use_gpu=True,
                        batch_size=max(4, self.batch_size // 2),
                        embedding_model="gpu_optimized"
                    )
                else:
                    vector_manager = VectorStoreManager(
                        db_path=self.db_path,
                        use_gpu=False,
                        batch_size=self.batch_size * 2,
                        embedding_model="lightweight" if self.ocr_mode == "lightweight" else "cpu_optimized"
                    )
                
                try:
                    vector_manager.store_text_content(document_id, analysis_results['text_pages'])
                    print(f"✓ Stored text from {len(analysis_results['text_pages'])} pages")
                finally:
                    if hasattr(vector_manager, 'cleanup'):
                        vector_manager.cleanup()
                    self._clear_memory()
            
            elif analysis_results['text_pages']:
                print("\n5. Vector store not available, skipping text storage")
            else:
                print("\n5. No text content found to store")
            self.db_manager.update_document_status(document_id, "completed")
            
            print("\n" + "="*60)
            print("PDF PROCESSING COMPLETE!")
            print("="*60)
            print(f"Document ID: {document_id}")
            print(f"Database: {self.db_path}")
            print(f"Analysis: {self.analysis_file}")
            print(f"Images: {self.image_dir}")
            data_counts = self.db_manager.get_document_data_counts(document_id)
            print(f"Data stored:")
            print(f"  - Tables: {data_counts['tables']}")
            print(f"  - Formulas: {data_counts['formulas']}")
            print(f"  - Text chunks: {data_counts['text_chunks']}")
            
            return True
            
        except Exception as e:
            print(f"\nError during processing: {str(e)}")
            if 'document_id' in locals() and document_id:
                self.db_manager.update_document_status(document_id, "failed")
            return False
        finally:
            self._clear_memory()
    
    def start_query_interface(self, document_id=None):
        """
        Start interactive query interface with optional document filtering
        """
        print("\n" + "="*60)
        print("STARTING RAG QUERY INTERFACE")
        print("="*60)
        
        try:
            rag = RAGPipeline(db_path=self.db_path)
            
            print("\nRAG system ready! You can now ask questions about the processed PDF(s).")
            print(f"Batch Size: {self.batch_size}")
            
            if document_id:
                print(f"Filtering queries to document ID: {document_id}")
            
            print("\nExample queries:")
            print("- 'Show me tables with financial data'")
            print("- 'Find formulas related to velocity'") 
            print("- 'Explain the concept mentioned on page 5'")
            print("- 'What mathematical equations are in this document?'")
            print("\nType 'quit' to exit.")
            
            while True:
                print("\n" + "-"*40)
                query = input("Enter your query: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                if not query:
                    continue
                
                try:
                    response = rag.process_query(query)
                    print(f"\nResponse:\n{response}")
                except Exception as e:
                    print(f"Error processing query: {str(e)}")
                finally:
                    self._clear_memory()
            
            print("\nQuery session ended.")
            
        except Exception as e:
            print(f"Error starting RAG interface: {str(e)}")
        finally:
            self._clear_memory()
    
    def list_documents(self):
        """List all documents in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, filename, file_path, file_size, page_count, processed_at, processing_status
        FROM documents
        ORDER BY processed_at DESC
        ''')
        
        documents = cursor.fetchall()
        conn.close()
        
        if not documents:
            print("No documents found in database.")
            return
        
        print("\nDocuments in database:")
        print("-" * 80)
        print(f"{'ID':<4} {'Filename':<30} {'Pages':<6} {'Status':<12} {'Processed':<20}")
        print("-" * 80)
        
        for doc in documents:
            doc_id, filename, file_path, file_size, page_count, processed_at, status = doc
            filename_short = filename[:27] + "..." if len(filename) > 30 else filename
            processed_short = processed_at[:16] if processed_at else "Never"
            print(f"{doc_id:<4} {filename_short:<30} {page_count:<6} {status:<12} {processed_short:<20}")
        
        return documents

def main():
    import sqlite3
    
    parser = argparse.ArgumentParser(description='Smart PDF RAG Processing Pipeline with Document Tracking')
    parser.add_argument('pdf_path', nargs='?', help='Path to PDF file to process')
    parser.add_argument('--output_dir', default='pdf_processing', help='Output directory')
    parser.add_argument('--db_path', default='rag_database.db', help='SQLite database path')
    parser.add_argument('--dpi', type=int, default=300, help='DPI for PDF to image conversion')
    parser.add_argument('--no_query', action='store_true', help='Skip query interface after processing')
    parser.add_argument('--list_docs', action='store_true', help='List all documents in database')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if document exists')
    
    parser.add_argument('--gpu_mode', choices=['auto', 'force', 'disable'], default='auto',
                       help='GPU usage mode')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for processing')
    parser.add_argument('--ocr_mode', choices=['lightweight', 'cpu_accurate', 'gpu_optimized'], 
                       default='lightweight', help='OCR mode')
    parser.add_argument('--low_memory', action='store_true',
                       help='Use low memory settings')
    
    args = parser.parse_args()
    
    if args.low_memory:
        args.batch_size = min(args.batch_size, 2)
        if args.ocr_mode == 'gpu_optimized':
            print("Low memory mode: switching to lightweight OCR")
            args.ocr_mode = 'lightweight'
        if args.gpu_mode == 'auto':
            args.gpu_mode = 'disable'
            print("Low memory mode: disabling GPU")
    
    orchestrator = SmartPDFRAGOrchestrator(
        output_dir=args.output_dir,
        db_path=args.db_path,
        gpu_mode=args.gpu_mode,
        batch_size=args.batch_size,
        ocr_mode=args.ocr_mode,
        force_reprocess=args.force
    )
    
    if args.list_docs:
        documents = orchestrator.list_documents()
        if not args.no_query and documents:
            print(f"\nTo query specific document, use: --document_id <ID>")
        return
    
    if not args.pdf_path:
        print("Error: PDF path required unless using --list_docs")
        print("Use --help for usage information")
        sys.exit(1)
    
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    print("Smart PDF RAG Pipeline with Document Tracking")
    print(f"PDF: {args.pdf_path}")
    print(f"Force reprocess: {args.force}")
    
    success = orchestrator.process_pdf(args.pdf_path, args.dpi)
    
    if success and not args.no_query:
        orchestrator.start_query_interface()
    elif success:
        print(f"\nProcessing complete. To start queries later, run:")
        print(f"python {__file__} --list_docs")
        print(f"python rag_pipeline.py")
    orchestrator._clear_memory()
if __name__ == "__main__":
    main()
