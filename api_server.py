"""
Updated API Server with improved error handling for Excel files and validation
"""

import os
import tempfile
import logging
from typing import Dict, Any, Optional, List
import json

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import our modules
from preprocessing_module import preprocess_file
from document_classifier import classify_document
from gpt_data_extractor import extract_financial_data
from validation_formatter import validate_and_format_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('api_server')

# Create FastAPI app
app = FastAPI(
    title="Real Estate NOI Analyzer - Data Extraction API",
    description="API for extracting financial data from real estate documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define response models
class FinancialsModel(BaseModel):
    rental_income: Optional[float] = None
    laundry_income: Optional[float] = None
    parking_income: Optional[float] = None
    other_revenue: Optional[float] = None
    total_revenue: Optional[float] = None
    repairs_maintenance: Optional[float] = None
    utilities: Optional[float] = None
    property_management_fees: Optional[float] = None
    property_taxes: Optional[float] = None
    insurance: Optional[float] = None
    admin_office_costs: Optional[float] = None
    marketing_advertising: Optional[float] = None
    total_expenses: Optional[float] = None
    net_operating_income: Optional[float] = None

class ExtractedDataModel(BaseModel):
    document_type: str
    period: str
    financials: FinancialsModel
    warnings: List[str] = Field(default_factory=list)

# API key validation
def get_api_key(x_api_key: str = Header(...)):
    """
    Validate the API key
    
    Args:
        x_api_key: API key from header
        
    Returns:
        API key if valid
    """
    expected_api_key = os.environ.get("API_KEY")
    if not expected_api_key:
        # If no API key is set in environment, allow all requests (development mode)
        return x_api_key
    
    if x_api_key != expected_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return x_api_key

def get_openai_api_key(x_openai_api_key: Optional[str] = Header(None)):
    """
    Get the OpenAI API key from header or environment
    
    Args:
        x_openai_api_key: OpenAI API key from header
        
    Returns:
        OpenAI API key
    """
    # First try to get from header
    if x_openai_api_key:
        return x_openai_api_key
    
    # Then try to get from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise HTTPException(
            status_code=400, 
            detail="OpenAI API key not provided. Please set OPENAI_API_KEY environment variable or provide x-openai-api-key header."
        )
    
    return openai_api_key

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Real Estate NOI Analyzer Data Extraction API"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/extract", response_model=ExtractedDataModel)
async def extract_data(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key),
    openai_api_key: str = Depends(get_openai_api_key)
):
    """
    Extract financial data from a document
    
    Args:
        file: Uploaded document (PDF, Excel, CSV, TXT)
        api_key: API key for authentication
        openai_api_key: OpenAI API key
        
    Returns:
        Extracted financial data
    """
    logger.info(f"Processing file: {file.filename}")
    
    # Check file extension
    file_ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
    if file_ext not in ['pdf', 'xlsx', 'xls', 'csv', 'txt']:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
    
    # Save uploaded file to temp directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        # Step 1: Preprocess the file
        logger.info("Preprocessing file")
        try:
            preprocessed_data = preprocess_file(temp_file_path)
            logger.info("Preprocessing completed successfully")
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error preprocessing file: {str(e)}")
        
        # Step 2: Classify the document
        logger.info("Classifying document")
        combined_text = preprocessed_data.get('content', {}).get('combined_text', '')
        if not combined_text:
            # Try to get text from different sources based on file type
            if file_ext in ['pdf', 'txt']:
                combined_text = '\n'.join([page.get('content', '') for page in preprocessed_data.get('content', {}).get('text', [])])
            elif file_ext in ['xlsx', 'xls', 'csv']:
                combined_text = preprocessed_data.get('content', {}).get('text_representation', '')
        
        try:
            classification_result = classify_document(combined_text, openai_api_key)
            logger.info(f"Classification result: {classification_result}")
            
            # Ensure document_type and period are valid strings
            document_type = classification_result.get('document_type', 'Unknown')
            if document_type is None:
                document_type = 'Unknown'
                
            # Extract period from filename if not found by classifier
            period = classification_result.get('period', '')
            if not period:
                # Try to extract from filename (e.g., Actual_Mar_2025.xlsx)
                filename_parts = os.path.splitext(file.filename)[0].split('_')
                if len(filename_parts) >= 2:
                    # Look for month abbreviations in filename parts
                    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                    year_pattern = r'20\d{2}'
                    
                    month = None
                    year = None
                    
                    for part in filename_parts:
                        # Check for month
                        for m in months:
                            if m.lower() in part.lower():
                                month = m
                                break
                        
                        # Check for year (2020-2099)
                        import re
                        year_match = re.search(year_pattern, part)
                        if year_match:
                            year = year_match.group(0)
                    
                    if month and year:
                        period = f"{month} {year}"
                    elif year:
                        period = year
            
            # If still no period, use a default
            if not period:
                period = "Unknown Period"
                
        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            document_type = "Unknown"
            period = "Unknown Period"
        
        # Step 3: Extract financial data using GPT
        logger.info("Extracting financial data")
        try:
            extracted_data = extract_financial_data(combined_text, document_type, period, openai_api_key)
            logger.info("Financial data extraction completed")
        except Exception as e:
            logger.error(f"Financial data extraction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error extracting financial data: {str(e)}")
        
        # Step 4: Validate and format the data
        logger.info("Validating and formatting data")
        try:
            formatted_data, warnings = validate_and_format_data(extracted_data)
            logger.info(f"Validation completed with {len(warnings)} warnings")
            
            # Ensure document_type and period are valid strings
            if formatted_data.get('document_type') is None:
                formatted_data['document_type'] = "Unknown"
            if formatted_data.get('period') is None:
                formatted_data['period'] = "Unknown Period"
                
            # Add warnings to the response
            formatted_data['warnings'] = warnings
            
            return formatted_data
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error validating data: {str(e)}")
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
