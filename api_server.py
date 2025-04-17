"""
Batch API Server for NOI Data Extraction
This server provides endpoints for extracting financial data from documents,
including a batch endpoint for processing multiple files at once.
"""

import os
import tempfile
import logging
from typing import List, Dict, Any, Optional, Union
import json

from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import openai

# Import preprocessing and extraction modules
from preprocessing_module import preprocess_file
from document_classifier import classify_document, extract_period_from_filename
from gpt_data_extractor import extract_financial_data
from validation_formatter import validate_and_format_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('noi_extraction_api')

# Initialize FastAPI app
app = FastAPI(
    title="NOI Data Extraction API",
    description="API for extracting financial data from real estate documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get API key from environment variable
API_KEY = os.environ.get("API_KEY", "your-api-key")

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai.api_key = OPENAI_API_KEY
else:
    logger.warning("OPENAI_API_KEY not set. GPT extraction will not work.")

# Response models
class ExtractedDataModel(BaseModel):
    document_type: str
    period: str
    financials: Dict[str, Any]

class BatchExtractedDataModel(BaseModel):
    results: List[Dict[str, Any]]
    consolidated_data: Dict[str, Any]

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {"message": "NOI Data Extraction API is running"}

@app.post("/extract", response_model=ExtractedDataModel)
async def extract_data(file: UploadFile = File(...), api_key: str = Header(None)):
    """
    Extract financial data from a single document
    
    Args:
        file: Uploaded file (PDF, Excel, CSV, TXT)
        api_key: API key for authentication
        
    Returns:
        Extracted financial data
    """
    # Validate API key
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Get file extension
    file_ext = file.filename.split('.')[-1].lower()
    logger.info(f"Processing file: {file.filename} (type: {file_ext})")
    
    # Save uploaded file to temp directory
    file_content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    
    try:
        # Step 1: Preprocess the file
        logger.info(f"Preprocessing file: {file.filename}")
        preprocessed_data = preprocess_file(temp_file_path)
        
        # Step 2: Classify the document
        logger.info(f"Classifying document: {file.filename}")
        doc_type, period = classify_document(preprocessed_data)
        
        # If period not found in content, try to extract from filename
        if not period:
            period = extract_period_from_filename(file.filename)
            logger.info(f"Extracted period from filename: {period}")
        
        # Step 3: Extract financial data
        logger.info(f"Extracting financial data: {file.filename}")
        extracted_data = extract_financial_data(preprocessed_data, doc_type)
        
        # Step 4: Validate and format data
        logger.info(f"Validating and formatting data: {file.filename}")
        validated_data = validate_and_format_data(extracted_data)
        
        # Ensure document_type and period are never null
        doc_type = doc_type or "Actual"  # Default to Actual if not determined
        period = period or "Unknown Period"
        
        # Return extracted data
        result = {
            "document_type": doc_type,
            "period": period,
            "financials": validated_data
        }
        
        logger.info(f"Successfully extracted data from {file.filename}")
        return result
    
    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_file_path)
        except:
            pass

@app.post("/extract-batch", response_model=BatchExtractedDataModel)
async def extract_batch_data(files: List[UploadFile] = File(...), api_key: str = Header(None)):
    """
    Extract financial data from multiple documents
    
    Args:
        files: List of uploaded files (PDF, Excel, CSV, TXT)
        api_key: API key for authentication
        
    Returns:
        List of extracted financial data and consolidated data structure
    """
    # Validate API key
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    logger.info(f"Processing batch of {len(files)} files")
    
    results = []
    for file in files:
        # Get file extension
        file_ext = file.filename.split('.')[-1].lower()
        logger.info(f"Processing file: {file.filename} (type: {file_ext})")
        
        # Save uploaded file to temp directory
        file_content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Step 1: Preprocess the file
            logger.info(f"Preprocessing file: {file.filename}")
            preprocessed_data = preprocess_file(temp_file_path)
            
            # Step 2: Classify the document
            logger.info(f"Classifying document: {file.filename}")
            doc_type, period = classify_document(preprocessed_data)
            
            # If period not found in content, try to extract from filename
            if not period:
                period = extract_period_from_filename(file.filename)
                logger.info(f"Extracted period from filename: {period}")
            
            # Step 3: Extract financial data
            logger.info(f"Extracting financial data: {file.filename}")
            extracted_data = extract_financial_data(preprocessed_data, doc_type)
            
            # Step 4: Validate and format data
            logger.info(f"Validating and formatting data: {file.filename}")
            validated_data = validate_and_format_data(extracted_data)
            
            # Ensure document_type and period are never null
            doc_type = doc_type or "Actual"  # Default to Actual if not determined
            period = period or "Unknown Period"
            
            # Add result
            result = {
                "document_type": doc_type,
                "period": period,
                "financials": validated_data,
                "filename": file.filename
            }
            results.append(result)
            logger.info(f"Successfully extracted data from {file.filename}")
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            # Add error result
            results.append({
                "document_type": "Error",
                "period": "Error",
                "error": str(e),
                "filename": file.filename
            })
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass
    
    # Create consolidated data structure
    consolidated_data = create_consolidated_data(results)
    
    return {
        "results": results,
        "consolidated_data": consolidated_data
    }

def create_consolidated_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create consolidated data structure from multiple document results
    
    Args:
        results: List of extracted data from multiple documents
        
    Returns:
        Consolidated data structure for NOI comparisons
    """
    # Initialize data structure
    data = {}
    metadata = {
        "document_count": len(results),
        "periods": [],
        "has_budget": False,
        "has_prior_year": False
    }
    
    # Process each result
    for result in results:
        if "error" in result:
            continue
        
        doc_type = result["document_type"].lower()
        period = result["period"].replace(" ", "_")
        financials = result["financials"]
        
        # Add period to metadata if not already present
        if period not in metadata["periods"]:
            metadata["periods"].append(period)
        
        # Initialize period in data if not exists
        if period not in data:
            data[period] = {}
        
        # Categorize document
        category = None
        if "budget" in doc_type:
            category = "budget"
            metadata["has_budget"] = True
        elif "prior year" in doc_type or "2024" in period:
            category = "prior_year"
            metadata["has_prior_year"] = True
        else:
            category = "actual"
        
        # Add data to structure
        data[period][category] = financials
    
    # Create structure expected by calculate_noi_comparisons()
    consolidated_data = {
        "current_month": None,
        "prior_month": None,
        "budget": None,
        "prior_year": None
    }
    
    # Find current and prior months
    sorted_periods = sorted(metadata["periods"])
    if len(sorted_periods) > 0:
        current_period = sorted_periods[-1]
        if "actual" in data.get(current_period, {}):
            consolidated_data["current_month"] = format_for_noi_comparison(data[current_period]["actual"])
        
        # Find budget for current period
        if "budget" in data.get(current_period, {}):
            consolidated_data["budget"] = format_for_noi_comparison(data[current_period]["budget"])
        
        # Find prior year for current period
        if "prior_year" in data.get(current_period, {}):
            consolidated_data["prior_year"] = format_for_noi_comparison(data[current_period]["prior_year"])
        
        # Find prior month
        if len(sorted_periods) > 1:
            prior_period = sorted_periods[-2]
            if "actual" in data.get(prior_period, {}):
                consolidated_data["prior_month"] = format_for_noi_comparison(data[prior_period]["actual"])
    
    return {
        "data": data,
        "metadata": metadata,
        "consolidated_data": consolidated_data
    }

def format_for_noi_comparison(financials: Dict[str, Any]) -> Dict[str, float]:
    """
    Format financial data for NOI comparison
    
    Args:
        financials: Financial data from extraction
        
    Returns:
        Formatted data with revenue, expenses, and NOI
    """
    # Extract the relevant values with defaults of 0 for missing values
    revenue = financials.get('total_revenue', 0)
    expenses = financials.get('total_expenses', 0)
    noi = financials.get('net_operating_income', 0)
    
    # If NOI is not provided but we have revenue and expenses, calculate it
    if noi == 0 and revenue != 0 and expenses != 0:
        noi = revenue - expenses
    
    # Format according to the expected structure
    formatted_data = {
        "revenue": revenue,
        "expenses": expenses,
        "noi": noi
    }
    
    return formatted_data

if __name__ == "__main__":
    # Run the server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
