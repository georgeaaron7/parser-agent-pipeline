import sqlite3
import json
from typing import List, Dict, Any, Optional
import re
from sentence_transformers import SentenceTransformer

try:
    from vector_store_manager import VectorStoreManager
    UPDATED_VECTOR_STORE = True
except ImportError:
    from vector_store_manager import VectorStoreManager
    UPDATED_VECTOR_STORE = False
    print("Using original vector store manager")

class RAGPipeline:
    def __init__(self, db_path="rag_database.db", qdrant_host="localhost", qdrant_port=6333):
        self.db_path = db_path
        if UPDATED_VECTOR_STORE:
            self.vector_manager = VectorStoreManager(
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                db_path=db_path,
                use_gpu=False,  
                embedding_model="cpu_optimized"
            )
        else:
            self.vector_manager = VectorStoreManager(
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                db_path=db_path
            )
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def enhance_query(self, query: str) -> Dict[str, Any]:
        """Enhanced query analysis with answer generation intents"""
        query_lower = query.lower()
        intent_indicators = {
            'summary': ['summary', 'summarize', 'summarise', 'overview', 'main points'],
            'table': ['table', 'data', 'values', 'statistics', 'numbers', 'chart', 'column', 'row'],
            'formula': ['formula', 'equation', 'calculate', 'computation', 'math', 'derive', 'solve'],
            'text': ['explain', 'describe', 'what is', 'definition', 'concept', 'theory', 'information'],
            'comparison': ['compare', 'difference', 'versus', 'vs', 'similar', 'different'],
            'example': ['example', 'instance', 'case', 'demonstration', 'sample'],
            'howto': ['how', 'how to', 'steps', 'process', 'method']
        }
        detected_intents = []
        for intent, keywords in intent_indicators.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_intents.append(intent)
        if not detected_intents:
            detected_intents = ['table', 'formula', 'text']
        variables = re.findall(r'\b[a-zA-Z][a-zA-Z0-9]*\b', query)
        numbers = re.findall(r'\d+\.?\d*', query)
        enhanced_query = {
            'original_query': query,
            'intents': detected_intents,
            'variables': variables,
            'numbers': numbers,
            'keywords': query.split()
        }
        return enhanced_query
    
    def search_tables(self, query: str, limit: int = 3, document_id: Optional[int] = None) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if document_id is not None:
                cursor.execute('''
                SELECT id, document_id, page_number, source_file, table_data, description, columns
                FROM tables
                WHERE (description LIKE ? OR columns LIKE ?) AND document_id = ?
                ORDER BY page_number
                LIMIT ?
                ''', (f'%{query}%', f'%{query}%', document_id, limit))
            else:
                cursor.execute('''
                SELECT id, document_id, page_number, source_file, table_data, description, columns
                FROM tables
                WHERE description LIKE ? OR columns LIKE ?
                ORDER BY page_number
                LIMIT ?
                ''', (f'%{query}%', f'%{query}%', limit))
            results = []
            for row in cursor.fetchall():
                table_data = json.loads(row[4]) if row[4] else []
                columns = json.loads(row[6]) if row[6] else []
                results.append({
                    'id': row[0],
                    'document_id': row[1],
                    'page_number': row[2],
                    'source_file': row[3],
                    'table_data': table_data,
                    'description': row[5],
                    'columns': columns,
                    'type': 'table'
                })
            conn.close()
            return results
        except Exception as e:
            print(f"Error searching tables: {str(e)}")
            return []
    
    def search_formulas(self, query: str, limit: int = 3, document_id: Optional[int] = None) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            if document_id is not None:
                cursor.execute('''
                SELECT id, document_id, page_number, source_file, original_formula, parsed_formula, 
                       formula_type, variables, description, executable_code
                FROM formulas
                WHERE (original_formula LIKE ? OR description LIKE ? OR variables LIKE ?) AND document_id = ?
                ORDER BY page_number
                LIMIT ?
                ''', (f'%{query}%', f'%{query}%', f'%{query}%', document_id, limit))
            else:
                cursor.execute('''
                SELECT id, document_id, page_number, source_file, original_formula, parsed_formula, 
                       formula_type, variables, description, executable_code
                FROM formulas
                WHERE original_formula LIKE ? OR description LIKE ? OR variables LIKE ?
                ORDER BY page_number
                LIMIT ?
                ''', (f'%{query}%', f'%{query}%', f'%{query}%', limit))
            results = []
            for row in cursor.fetchall():
                variables = json.loads(row[7]) if row[7] else []
                results.append({
                    'id': row[0],
                    'document_id': row[1],
                    'page_number': row[2],
                    'source_file': row[3],
                    'original_formula': row[4],
                    'parsed_formula': row[5],
                    'formula_type': row[6],
                    'variables': variables,
                    'description': row[8],
                    'executable_code': row[9],
                    'type': 'formula'
                })
            conn.close()
            return results
        except Exception as e:
            print(f"Error searching formulas: {str(e)}")
            return []
    
    def search_text(self, query: str, limit: int = 5, document_id: Optional[int] = None) -> List[Dict]:
        if UPDATED_VECTOR_STORE:
            return self.vector_manager.search_similar_text(query, limit, document_id)
        else:
            return self.vector_manager.search_similar_text(query, limit)
    
    def get_all_document_content(self, document_id: Optional[int] = None) -> Dict[str, List]:
        """Get all content for document summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            context = {'tables': [], 'formulas': [], 'text': []}
            if document_id:
                cursor.execute('SELECT * FROM tables WHERE document_id = ? ORDER BY page_number', (document_id,))
            else:
                cursor.execute('SELECT * FROM tables ORDER BY document_id, page_number')
            
            for row in cursor.fetchall():
                table_data = json.loads(row[4]) if row[4] else []
                context['tables'].append({
                    'document_id': row[1],
                    'page_number': row[2],
                    'description': row[5],
                    'table_data': table_data,
                    'type': 'table'
                })
            if document_id:
                cursor.execute('SELECT * FROM formulas WHERE document_id = ? ORDER BY page_number', (document_id,))
            else:
                cursor.execute('SELECT * FROM formulas ORDER BY document_id, page_number')
            
            for row in cursor.fetchall():
                variables = json.loads(row[7]) if row[7] else []
                context['formulas'].append({
                    'document_id': row[1],
                    'page_number': row[2],
                    'original_formula': row[4],
                    'description': row[8],
                    'variables': variables,
                    'type': 'formula'
                })
            if document_id:
                cursor.execute('SELECT * FROM text_content WHERE document_id = ? ORDER BY page_number, chunk_index', (document_id,))
            else:
                cursor.execute('SELECT * FROM text_content ORDER BY document_id, page_number, chunk_index')
            
            for row in cursor.fetchall():
                context['text'].append({
                    'document_id': row[1],
                    'page_number': row[2],
                    'text_content': row[4],
                    'score': 1.0,
                    'chunk_index': row[6]
                })
            
            conn.close()
            return context
            
        except Exception as e:
            print(f"Error getting document content: {e}")
            return {'tables': [], 'formulas': [], 'text': []}
    
    def execute_formula(self, formula_code: str, variables: Dict[str, float]) -> Optional[float]:
        try:
            safe_dict = {
                '__builtins__': {},
                'math': __import__('math'),
                'numpy': __import__('numpy'),
                'np': __import__('numpy'),
            }
            safe_dict.update(variables)
            exec(formula_code, safe_dict)
            if 'calculate' in safe_dict:
                if variables:
                    return safe_dict['calculate'](**variables)
                else:
                    return safe_dict['calculate']()
            return None
        except Exception as e:
            print(f"Error executing formula: {str(e)}")
            return None
    
    def get_document_info(self, document_id: int) -> Optional[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            SELECT id, filename, file_path, page_count, processed_at, processing_status
            FROM documents
            WHERE id = ?
            ''', (document_id,))
            result = cursor.fetchone()
            conn.close()
            if result:
                return {
                    'id': result[0],
                    'filename': result[1],
                    'file_path': result[2],
                    'page_count': result[3],
                    'processed_at': result[4],
                    'processing_status': result[5]
                }
            return None
        except Exception as e:
            print(f"Error getting document info: {e}")
            return None
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text for summaries"""
        important_terms = [
            'energy', 'mass', 'equation', 'formula', 'velocity', 'acceleration', 
            'force', 'power', 'Einstein', 'physics', 'calculation', 'theory',
            'experiment', 'analysis', 'data', 'result', 'conclusion', 'example'
        ]
        
        text_lower = text.lower()
        found_terms = []
        
        for term in important_terms:
            if term in text_lower:
                found_terms.append(term)
        words = text.split()
        capitalized = [word.strip('.,!?()[]{}') for word in words if len(word) > 3 and word[0].isupper()]
        all_terms = list(set(found_terms + capitalized[:10]))
        return all_terms[:8]
    
    def _create_document_summary(self, context: Dict[str, List], document_id: Optional[int] = None) -> str:
        """Create a comprehensive document summary"""
        summary_parts = []
        if document_id:
            doc_info = self.get_document_info(document_id)
            if doc_info:
                summary_parts.append(f"Summary of '{doc_info['filename']}':")
            else:
                summary_parts.append(f"Summary of Document {document_id}:")
        else:
            summary_parts.append("Document Summary:")
        text_content = context.get('text', [])
        if text_content:
            all_text = ' '.join([t['text_content'] for t in text_content[:10]])
            key_phrases = self._extract_key_phrases(all_text)
            summary_parts.append(f"\nMain Content:")
            summary_parts.append(f"This document contains {len(text_content)} text sections covering topics including: {', '.join(key_phrases[:5])}.")
            summary_parts.append(f"\nKey Points:")
            for i, text in enumerate(text_content[:3]):
                content = text['text_content']
                if len(content) > 200:
                    content = content[:200] + "..."
                summary_parts.append(f"• {content}")
        tables = context.get('tables', [])
        if tables:
            summary_parts.append(f"\nTables and Data:")
            summary_parts.append(f"The document contains {len(tables)} table(s):")
            for table in tables:
                summary_parts.append(f"• Page {table['page_number']}: {table['description']}")
        formulas = context.get('formulas', [])
        if formulas:
            summary_parts.append(f"\nFormulas and Equations:")
            summary_parts.append(f"The document contains {len(formulas)} mathematical formula(s):")
            for formula in formulas:
                summary_parts.append(f"• Page {formula['page_number']}: {formula['original_formula']} - {formula['description']}")
        
        if not text_content and not tables and not formulas:
            summary_parts.append("\nNo content found in the document.")
        
        return "\n".join(summary_parts)
    
    def _create_definition_answer(self, query: str, context: Dict[str, List]) -> str:
        """Create definition-style answers"""
        text_content = context.get('text', [])
        if not text_content:
            return "I couldn't find relevant definitions in the document."
        
        query_words = query.lower().replace('what is', '').replace('define', '').strip().split()
        relevant_sentences = []
        
        for text in text_content:
            sentences = text['text_content'].split('. ')
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(word in sentence_lower for word in query_words) and len(sentence) > 20:
                    relevant_sentences.append(sentence.strip())
        
        if relevant_sentences:
            answer = f"Based on the document:\n\n{relevant_sentences[0]}"
            if len(relevant_sentences) > 1:
                answer += f"\n\nAdditionally: {relevant_sentences[1]}"
            return answer
        else:
            return f"The document mentions: {text_content[0]['text_content'][:300]}..."
    
    def _create_how_to_answer(self, query: str, context: Dict[str, List]) -> str:
        """Create how-to style answers"""
        formulas = context.get('formulas', [])
        text_content = context.get('text', [])
        
        answer_parts = ["Here's how to approach this based on the document:"]
        
        if formulas:
            answer_parts.append(f"\nRelevant Formula:")
            for formula in formulas[:2]:
                answer_parts.append(f"• {formula['original_formula']} ({formula['description']})")
        
        if text_content:
            for text in text_content:
                content = text['text_content']
                if any(word in content.lower() for word in ['step', 'first', 'then', 'next', 'calculate']):
                    answer_parts.append(f"\nProcess:")
                    answer_parts.append(content[:400] + ("..." if len(content) > 400 else ""))
                    break
        
        return "\n".join(answer_parts)
    
    def _create_explanation_answer(self, query: str, context: Dict[str, List]) -> str:
        """Create explanation-style answers"""
        text_content = context.get('text', [])
        
        if not text_content:
            return "I couldn't find explanatory content in the document."
        
        answer = "The document explains:\n\n"
        
        explanatory_content = []
        for text in text_content:
            content = text['text_content']
            if any(word in content.lower() for word in ['because', 'reason', 'explains', 'theory', 'principle']):
                explanatory_content.append(content)
        
        if explanatory_content:
            answer += explanatory_content[0][:500] + ("..." if len(explanatory_content[0]) > 500 else "")
        else:
            answer += text_content[0]['text_content'][:500] + ("..." if len(text_content[0]['text_content']) > 500 else "")
        
        return answer
    
    def _create_example_answer(self, query: str, context: Dict[str, List]) -> str:
        """Create example-focused answers"""
        text_content = context.get('text', [])
        tables = context.get('tables', [])
        
        answer_parts = ["Here are examples from the document:"]
        
        for text in text_content:
            content = text['text_content']
            if any(word in content.lower() for word in ['example', 'instance', 'case', 'suppose']):
                answer_parts.append(f"\nExample from page {text['page_number']}:")
                answer_parts.append(content[:300] + ("..." if len(content) > 300 else ""))
                break
        
        if tables:
            answer_parts.append(f"\nData Examples:")
            for table in tables[:1]:
                if table['table_data']:
                    answer_parts.append(f"From page {table['page_number']}:")
                    for i, row in enumerate(table['table_data'][:3]):
                        answer_parts.append(f"  {', '.join(str(cell) for cell in row)}")
        
        return "\n".join(answer_parts)
    
    def _create_general_answer(self, query: str, context: Dict[str, List]) -> str:
        """Create general answers"""
        text_content = context.get('text', [])
        
        if not text_content:
            return "I couldn't find relevant information in the document."
        
        query_words = set(query.lower().split())
        best_match = None
        best_score = 0
        
        for text in text_content:
            content_words = set(text['text_content'].lower().split())
            overlap = len(query_words.intersection(content_words))
            if overlap > best_score:
                best_score = overlap
                best_match = text
        
        if best_match:
            answer = f"Regarding your question:\n\n{best_match['text_content']}"
            if len(answer) > 600:
                answer = answer[:600] + "..."
            return answer
        else:
            return f"Based on the document content:\n\n{text_content[0]['text_content'][:400]}..."
    
    def generate_response(self, query: str, context: Dict[str, List], document_id: Optional[int] = None) -> str:
        """Generate natural language response based on query type and context"""
        
        query_lower = query.lower()
        if any(word in query_lower for word in ['summary', 'summarize', 'summarise', 'overview']):
            return self._create_document_summary(context, document_id)
        if any(word in query_lower for word in ['what is', 'define', 'definition']):
            return self._create_definition_answer(query, context)
        elif any(word in query_lower for word in ['how', 'calculate', 'compute']):
            return self._create_how_to_answer(query, context)
        elif any(word in query_lower for word in ['why', 'reason', 'because']):
            return self._create_explanation_answer(query, context)
        elif any(word in query_lower for word in ['example', 'examples', 'instance']):
            return self._create_example_answer(query, context)
        else:
            return self._create_general_answer(query, context)
    
    def process_query(self, query: str, document_id: Optional[int] = None) -> str:
        """Process query with integrated answer generation"""
        print(f"Processing query: {query}")
        if document_id:
            print(f"Filtering to document ID: {document_id}")
        
        enhanced_query = self.enhance_query(query)
        print(f"Detected intents: {enhanced_query['intents']}")
        if 'summary' in enhanced_query['intents']:
            context = self.get_all_document_content(document_id)
        else:
            context = {}
            
            if 'table' in enhanced_query['intents']:
                context['tables'] = self.search_tables(query, document_id=document_id)
                print(f"Found {len(context['tables'])} relevant tables")
            
            if 'formula' in enhanced_query['intents']:
                context['formulas'] = self.search_formulas(query, document_id=document_id)
                print(f"Found {len(context['formulas'])} relevant formulas")
            
            if 'text' in enhanced_query['intents']:
                context['text'] = self.search_text(query, document_id=document_id)
                print(f"Found {len(context['text'])} relevant text chunks")
        response = self.generate_response(query, context, document_id)
        return response

if __name__ == "__main__":
    import sys
    rag = RAGPipeline()
    document_id = None
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        document_id = int(sys.argv[1])
        print(f"Filtering queries to document ID: {document_id}")
    print("RAG Pipeline with Answer Generation initialized. Type 'quit' to exit.")
    print("Ask questions about the processed PDF content.")
    while True:
        query = input("\nEnter your query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
        try:
            response = rag.process_query(query, document_id)
            print(f"\nResponse:\n{response}")
        except Exception as e:
            print(f"Error processing query: {str(e)}")
    print("RAG Pipeline session ended.")
