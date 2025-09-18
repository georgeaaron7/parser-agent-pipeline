# PDF RAG Processing Pipeline

A GPU-optimized Python pipeline for processing PDFs and creating a Retrieval-Augmented Generation (RAG) system. This tool extracts text, tables, and formulas from PDF documents, stores them in a searchable database, and provides an interactive query interface.

## Features

- **Multi-format Content Extraction**: Extracts text, tables, and mathematical formulas from PDF documents
- **GPU Optimization**: Supports GPU acceleration for faster processing (optional)
- **OCR Options**: Multiple OCR modes from lightweight to high-accuracy
- **Vector Search**: Semantic search using vector embeddings
- **Interactive Querying**: Built-in RAG system for natural language queries
- **Flexible Configuration**: Customizable processing parameters and memory settings

## Architecture

The pipeline consists of several interconnected components:

1. **PDF Converter** (`pdf_converter.py`): Converts PDF pages to images
2. **Content Detector** (`content_detector.py`): Identifies and extracts different content types
3. **Table Extractor** (`table_extractor.py`): Processes and structures table data
4. **Formula Processor** (`formula_processor.py`): Parses and processes mathematical formulas
5. **Vector Store Manager** (`vector_store_manager.py`): Manages vector embeddings and semantic search
6. **RAG Pipeline** (`rag_pipeline.py`): Provides query processing and response generation
7. **Main Orchestrator** (`main.py`): Coordinates the entire processing workflow

## Installation

### Prerequisites

- Python 3.8+
- [Tesseract OCR](https://tesseract-ocr.github.io/tessdoc/Installation.html) (for lightweight OCR mode)
- CUDA-compatible GPU (optional, for GPU acceleration)
- [Qdrant](https://qdrant.tech/) vector database (running locally or remotely)

### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install tesseract-ocr poppler-utils
```

**macOS:**
```bash
brew install tesseract poppler
```

**Windows:**
- Install [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- Install [Poppler](https://blog.alivate.com.au/poppler-windows/)

### Python Dependencies

```bash
# Clone the repository
git clone https://github.com/georgeaaron7/parser-agent-pipeline.git
cd parser-agent-pipeline

# Install Python dependencies
pip install -r requirements.txt
```

### Qdrant Setup

**Option 1: Docker (Recommended)**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Option 2: Local Installation**
Follow the [Qdrant installation guide](https://qdrant.tech/documentation/quick_start/)

## Quick Start

### Basic Usage

```bash
# Process a PDF with default settings
python main.py document.pdf

# Process with custom output directory
python main.py document.pdf --output_dir my_analysis

# Process without interactive query mode
python main.py document.pdf --no_query
```

### GPU-Optimized Processing

```bash
# Auto-detect and use GPU if available
python main.py document.pdf --gpu_mode auto --ocr_mode gpu_optimized

# Force GPU usage (will fail if GPU unavailable)
python main.py document.pdf --gpu_mode force --ocr_mode gpu_optimized --batch_size 8

# Disable GPU entirely
python main.py document.pdf --gpu_mode disable --ocr_mode lightweight
```

### Memory-Constrained Environments

```bash
# Low memory mode (reduces batch sizes, uses lightweight models)
python main.py document.pdf --low_memory

# Custom batch size for memory optimization
python main.py document.pdf --batch_size 2 --ocr_mode lightweight
```

## Command Line Options

### Required Arguments
- `pdf_path`: Path to the PDF file to process

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `pdf_processing` | Directory for output files |
| `--db_path` | `rag_database.db` | SQLite database path |
| `--dpi` | `300` | DPI for PDF to image conversion |
| `--no_query` | `False` | Skip interactive query interface |

### GPU & Performance Options

| Argument | Choices | Default | Description |
|----------|---------|---------|-------------|
| `--gpu_mode` | `auto`, `force`, `disable` | `auto` | GPU usage mode |
| `--batch_size` | Integer | `4` | Batch size for processing |
| `--ocr_mode` | `lightweight`, `cpu_accurate`, `gpu_optimized` | `lightweight` | OCR processing mode |
| `--low_memory` | Flag | `False` | Enable low memory settings |

### OCR Mode Details

- **lightweight**: Fast Tesseract OCR (CPU-only, minimal memory)
- **cpu_accurate**: EasyOCR on CPU (higher accuracy, more memory)
- **gpu_optimized**: EasyOCR on GPU (fastest with GPU, highest memory usage)

## Usage Examples

### Example 1: Research Paper Analysis

```bash
# Process an academic paper with high accuracy
python main.py research_paper.pdf --ocr_mode cpu_accurate --batch_size 6
```

After processing, query the content:
```
Enter your query: What are the main findings of this research?
Enter your query: Show me tables with experimental results
Enter your query: What formulas are used for calculating accuracy?
```

### Example 2: Financial Document Processing

```bash
# Process financial reports with GPU acceleration
python main.py annual_report.pdf --gpu_mode auto --ocr_mode gpu_optimized --output_dir financial_analysis
```

### Example 3: Batch Processing Multiple Documents

```bash
# Process multiple PDFs (run separately)
for pdf in documents/*.pdf; do
    python main.py "$pdf" --output_dir "analysis_$(basename "$pdf" .pdf)" --no_query
done

# Then query all processed documents
python rag_pipeline.py
```

### Example 4: Server/Headless Environment

```bash
# Process without GUI dependencies
python main.py document.pdf --no_query --ocr_mode lightweight --gpu_mode disable
```

## Interactive Query Interface

After processing, you can query the extracted content using natural language:

```
RAG Pipeline initialized. Type 'quit' to exit.
Ask questions about the processed PDF content.

Enter your query: What tables contain financial data?
Enter your query: Show me formulas related to machine learning
Enter your query: Summarize the methodology section
Enter your query: quit
```

### Query Types Supported

- **Table queries**: "Show tables with sales data", "What financial information is available?"
- **Formula queries**: "What mathematical formulas are present?", "Show calculation methods"
- **Text queries**: "Summarize the conclusions", "What is the methodology?"
- **Mixed queries**: "How are the results calculated based on the data tables?"

## Configuration

### Database Configuration

The system uses SQLite for structured data and Qdrant for vector storage:

- **SQLite**: Stores tables, formulas, and metadata
- **Qdrant**: Stores text embeddings for semantic search

### Vector Store Settings

You can configure vector storage in `vector_store_manager.py`:

```python
# GPU-optimized configuration
vector_manager = VectorStoreManager(
    qdrant_host="localhost",
    qdrant_port=6333,
    use_gpu=True,
    batch_size=8,
    embedding_model="gpu_optimized"
)
```

### OCR Engine Configuration

Modify OCR settings in `content_detector.py`:

```python
detector = ContentDetector(
    ocr_mode="cpu_accurate",
    use_gpu=False,
    batch_size=4
)
```

## Output Structure

After processing, the following files and directories are created:

```
pdf_processing/
├── images/                 # Converted PDF pages as images
│   ├── document_page_1.png
│   ├── document_page_2.png
│   └── ...
├── content_analysis.json   # Detailed analysis results
rag_database.db            # SQLite database with extracted content
```

### Content Analysis JSON Structure

```json
{
  "table_pages": [
    {
      "page_number": 1,
      "image_path": "pdf_processing/images/document_page_1.png",
      "tables": [...]
    }
  ],
  "formula_pages": [
    {
      "page_number": 2,
      "formulas": [...]
    }
  ],
  "text_pages": [
    {
      "page_number": 1,
      "text_content": "...",
      "confidence": 0.95
    }
  ]
}
```

## Troubleshooting

### Common Issues

**1. ModuleNotFoundError: No module named 'pdf2image'**
```bash
pip install pdf2image
# Also ensure poppler is installed (see installation section)
```

**2. Tesseract not found**
```bash
# Ubuntu/Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows: Add Tesseract to PATH or set TESSDATA_PREFIX
```

**3. GPU memory errors**
```bash
# Reduce batch size or disable GPU
python main.py document.pdf --batch_size 2 --gpu_mode disable
```

**4. Qdrant connection failed**
```bash
# Start Qdrant server
docker run -p 6333:6333 qdrant/qdrant

# Or modify connection settings in vector_store_manager.py
```

**5. Out of memory errors**
```bash
# Use low memory mode
python main.py document.pdf --low_memory

# Or customize settings
python main.py document.pdf --batch_size 1 --ocr_mode lightweight
```

### Performance Optimization Tips

1. **GPU Memory**: Start with smaller batch sizes (2-4) and increase gradually
2. **CPU Processing**: Use larger batch sizes (8-16) for CPU-only mode
3. **Memory Usage**: Monitor memory usage and adjust `--batch_size` accordingly
4. **OCR Speed**: Use `lightweight` mode for faster processing, `cpu_accurate` for better accuracy

### Debug Mode

For debugging, you can run individual components:

```bash
# Test PDF conversion only
python pdf_converter.py document.pdf

# Test content detection only
python content_detector.py pdf_processing/images/ --mode lightweight

# Test vector storage only
python vector_store_manager.py content_analysis.json
```

## API Usage

You can also use the components programmatically:

```python
from main import PDFRAGOrchestrator
from rag_pipeline import RAGPipeline

# Initialize orchestrator
orchestrator = PDFRAGOrchestrator(
    output_dir="my_processing",
    gpu_mode="auto",
    batch_size=4,
    ocr_mode="lightweight"
)

# Process PDF
success = orchestrator.process_pdf("document.pdf", dpi=300)

if success:
    # Initialize RAG pipeline for querying
    rag = RAGPipeline(db_path="rag_database.db")
    response = rag.process_query("What are the main findings?")
    print(response)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text recognition
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for advanced OCR capabilities
- [Qdrant](https://qdrant.tech/) for vector similarity search
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for text embeddings
- [SymPy](https://www.sympy.org/) for mathematical formula processing

---

For more detailed information about specific components, please refer to the source code documentation in each module.