import cv2
import numpy as np
import pandas as pd
import pytesseract
from PIL import Image
import sqlite3
import json
import os
from typing import List, Dict, Tuple
from database_manager import DatabaseManager

class TableExtractor:
    def __init__(self, db_path="rag_database.db"):
        self.db_path = db_path
        self.db_manager = DatabaseManager(db_path)
    
    def extract_table_from_image(self, image_path: str) -> List[List[str]]:
        """
        Extract table data from image using computer vision and OCR
        """
        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cell_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  
                    x, y, w, h = cv2.boundingRect(contour)
                    cell_contours.append((x, y, w, h))
            cell_contours.sort(key=lambda x: (x[1], x[0]))
            table_data = []
            current_row = []
            current_y = None
            
            for x, y, w, h in cell_contours:
                if current_y is None:
                    current_y = y
                if abs(y - current_y) > 20 and current_row:
                    table_data.append(current_row)
                    current_row = []
                    current_y = y
                cell_image = gray[y:y+h, x:x+w]
                cell_text = pytesseract.image_to_string(cell_image, config='--psm 8').strip()
                current_row.append(cell_text)
            if current_row:
                table_data.append(current_row)
            
            return table_data
            
        except Exception as e:
            print(f"Error extracting table from {image_path}: {str(e)}")
            return []
    
    def generate_table_description(self, table_data: List[List[str]]) -> str:
        """
        Generate a brief description of the table content
        """
        if not table_data:
            return "Empty table"
        rows = len(table_data)
        cols = len(table_data[0]) if table_data else 0
        headers = table_data[0] if table_data else []
        description_parts = [f"Table with {rows} rows and {cols} columns"]
        
        if headers:
            header_text = ", ".join([h for h in headers if h.strip()])
            if header_text:
                description_parts.append(f"Headers: {header_text}")
        if len(table_data) > 1:
            sample_data = []
            for i in range(1, min(4, len(table_data))):  
                row_text = ", ".join([cell for cell in table_data[i] if cell.strip()])
                if row_text:
                    sample_data.append(row_text[:100])  
            
            if sample_data:
                description_parts.append(f"Sample data: {' | '.join(sample_data)}")
        
        return ". ".join(description_parts)
    
    def store_table(self, document_id: int, page_number: int, source_file: str, table_data: List[List[str]]):
        """
        Store extracted table data in SQLite database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            table_json = json.dumps(table_data)
            description = self.generate_table_description(table_data)
            columns = json.dumps(table_data[0]) if table_data else json.dumps([])
            cursor.execute('''
            INSERT INTO tables (document_id, page_number, source_file, table_data, description, columns)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (document_id, page_number, source_file, table_json, description, columns))
            
            conn.commit()
            conn.close()
            
            print(f"Stored table from page {page_number} for document {document_id}")
            
        except Exception as e:
            print(f"Error storing table: {str(e)}")
    
    def process_table_pages(self, document_id: int, table_pages: List[Dict]):
        """
        Process all pages containing tables for a specific document
        """
        if not table_pages:
            print("No table pages to process")
            return
            
        print(f"Processing {len(table_pages)} table pages for document {document_id}")
        
        for page_info in table_pages:
            page_number = page_info['page']
            image_path = page_info['path']
            
            print(f"Extracting table from page {page_number}")
            table_data = self.extract_table_from_image(image_path)
            if table_data:
                source_file = os.path.basename(image_path)
                self.store_table(document_id, page_number, source_file, table_data)
            else:
                print(f"No table data extracted from page {page_number}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python updated_table_extractor.py <content_analysis.json>")
        sys.exit(1)
    with open(sys.argv[1], 'r') as f:
        analysis_results = json.load(f)
    document_id = 1  
    extractor = TableExtractor()
    extractor.process_table_pages(document_id, analysis_results['table_pages'])
    
