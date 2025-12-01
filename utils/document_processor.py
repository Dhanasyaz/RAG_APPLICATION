"""Document text extraction"""
import PyPDF2
import docx

def extract_text_from_pdf(file):
    """
    Extract text from PDF file
    
    Args:
        file: File-like object (BytesIO or uploaded file)
        
    Returns:
        str: Extracted text
    """
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_text_from_docx(file):
    """
    Extract text from DOCX file
    
    Args:
        file: File-like object (BytesIO or uploaded file)
        
    Returns:
        str: Extracted text
    """
    doc = docx.Document(file)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return text


def extract_text_from_txt(file):
    """
    Extract text from TXT file
    
    Args:
        file: File-like object (BytesIO or uploaded file)
        
    Returns:
        str: Extracted text
    """
    return file.read().decode('utf-8')


def process_document(file):
    """
    Process uploaded document based on file type
    
    Args:
        file: Streamlit UploadedFile object
        
    Returns:
        str: Extracted text, or None if unsupported type
    """
    file_type = file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        return extract_text_from_pdf(file)
    elif file_type == 'docx':
        return extract_text_from_docx(file)
    elif file_type == 'txt':
        return extract_text_from_txt(file)
    else:
        print(f"Unsupported file type: {file_type}")
        return None