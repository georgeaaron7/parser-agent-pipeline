import sqlite3
import json
import re
import os
import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr
import numpy as np
from typing import List, Dict, Any
from database_manager import DatabaseManager

class FormulaProcessor:
    def __init__(self, db_path="rag_database.db"):
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
    
    def normalize_formula(self, formula_text: str) -> str:
        """
        Normalize formula text for better parsing
        """
        replacements = {
            '×': '*',
            '÷': '/',
            '–': '-',
            '—': '-',
            '²': '**2',
            '³': '**3',
            '√': 'sqrt',
            '∫': 'integrate',
            '∑': 'sum',
            'π': 'pi',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'δ': 'delta',
            'ε': 'epsilon',
            'θ': 'theta',
            'λ': 'lambda',
            'μ': 'mu',
            'σ': 'sigma',
        }
        
        normalized = formula_text
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized
    
    def parse_formula(self, formula_text: str) -> Dict[str, Any]:
        """
        Parse mathematical formula and extract components
        """
        try:
            normalized = self.normalize_formula(formula_text)
            
            try:
                expr = parse_expr(normalized, transformations='all')
                parsed_successfully = True
            except:
                try:
                    expr = parse_latex(normalized)
                    parsed_successfully = True
                except:
                    expr = None
                    parsed_successfully = False
            
            result = {
                'original': formula_text,
                'normalized': normalized,
                'parsed_successfully': parsed_successfully,
                'sympy_expr': str(expr) if expr else None,
                'variables': [],
                'formula_type': 'unknown',
                'executable_code': None
            }
            
            if expr:
                variables = list(expr.free_symbols)
                result['variables'] = [str(var) for var in variables]
                result['formula_type'] = self.classify_formula(expr, normalized)
                result['executable_code'] = self.generate_executable_code(expr, variables)
            
            return result
            
        except Exception as e:
            return {
                'original': formula_text,
                'normalized': self.normalize_formula(formula_text),
                'parsed_successfully': False,
                'error': str(e),
                'sympy_expr': None,
                'variables': [],
                'formula_type': 'unknown',
                'executable_code': None
            }
    
    def classify_formula(self, expr, normalized_text: str) -> str:
        """
        Classify the type of mathematical formula
        """
        text_lower = normalized_text.lower()
        
        if 'integrate' in text_lower or '∫' in normalized_text:
            return 'integral'
        elif 'sum' in text_lower or '∑' in normalized_text:
            return 'summation'
        elif 'sqrt' in text_lower or '√' in normalized_text:
            return 'root'
        elif any(op in normalized_text for op in ['**', '^']):
            return 'power'
        elif any(op in normalized_text for op in ['sin', 'cos', 'tan', 'log', 'ln', 'exp']):
            return 'transcendental'
        elif '=' in normalized_text:
            return 'equation'
        elif any(op in normalized_text for op in ['+', '-', '*', '/']):
            return 'arithmetic'
        else:
            return 'algebraic'
    
    def generate_executable_code(self, expr, variables: List) -> str:
        """
        Generate executable Python code for the formula
        """
        try:
            var_names = [str(var) for var in variables]
            if not var_names:
                code = f"def calculate():\n    import math\n    import numpy as np\n    return {expr}"
            else:
                params = ", ".join(var_names)
                code = f"def calculate({params}):\n    import math\n    import numpy as np\n    return {expr}"
            
            return code
            
        except Exception as e:
            return f"# Error generating code: {str(e)}\n# Original expression: {expr}"
    
    def generate_formula_description(self, formula_data: Dict) -> str:
        """
        Generate human-readable description of the formula
        """
        parts = []
        
        if formula_data['formula_type'] != 'unknown':
            parts.append(f"Type: {formula_data['formula_type']}")
        
        if formula_data['variables']:
            vars_str = ", ".join(formula_data['variables'])
            parts.append(f"Variables: {vars_str}")
        
        if formula_data['parsed_successfully']:
            parts.append("Successfully parsed")
        else:
            parts.append("Parsing attempted")
        
        parts.append(f"Original: {formula_data['original'][:100]}...")
        
        return " | ".join(parts)
    
    def store_formula(self, document_id: int, page_number: int, source_file: str, formula_data: Dict):
        """
        Store formula in SQLite database with document tracking
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            description = self.generate_formula_description(formula_data)
            variables_json = json.dumps(formula_data['variables'])
            
            cursor.execute('''
            INSERT INTO formulas (
                document_id, page_number, source_file, original_formula, parsed_formula,
                formula_type, variables, description, executable_code
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document_id,
                page_number,
                source_file,
                formula_data['original'],
                formula_data['sympy_expr'],
                formula_data['formula_type'],
                variables_json,
                description,
                formula_data['executable_code']
            ))
            
            conn.commit()
            conn.close()
            
            print(f"Stored formula from page {page_number} for document {document_id}: {formula_data['formula_type']}")
            
        except Exception as e:
            print(f"Error storing formula: {str(e)}")
    
    def process_formula_pages(self, document_id: int, formula_pages: List[Dict]):
        """
        Process all pages containing formulas for a specific document
        """
        if not formula_pages:
            print("No formula pages to process")
            return
            
        print(f"Processing {len(formula_pages)} formula pages for document {document_id}")
        
        for page_info in formula_pages:
            page_number = page_info['page']
            image_path = page_info['path']
            formulas = page_info['formulas']
            
            print(f"Processing {len(formulas)} formulas from page {page_number}")
            
            for formula_info in formulas:
                formula_text = formula_info['formula']
                formula_data = self.parse_formula(formula_text)
                source_file = os.path.basename(image_path)
                self.store_formula(document_id, page_number, source_file, formula_data)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python updated_formula_processor.py <content_analysis.json>")
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        analysis_results = json.load(f)
    document_id = 1  
    
    processor = FormulaProcessor()
    processor.process_formula_pages(document_id, analysis_results['formula_pages'])
    
    print("Formula processing and storage complete!")
