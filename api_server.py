"""
Enhanced API Server with Multi-File Processing for Real Estate NOI Analyzer
Provides endpoints for document processing, data extraction, and multi-file analysis
with compatibility for existing NOI tool
"""

import os
import tempfile
import logging
from typing import Dict, Any, Optional, List, Union
import json
import re
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Header, Form, Body
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

class ComparisonModel(BaseModel):
    type: str  # "budget_vs_actual", "year_over_year", "month_over_month"
    period: str
    comparison_period: Optional[str] = None
    base_document: str
    comparison_document: str
    variances: Dict[str, Union[float, Dict[str, float]]]
    variance_percentages: Dict[str, Union[float, Dict[str, float]]]

class MultiFileResultModel(BaseModel):
    documents: List[ExtractedDataModel]
    comparisons: List[ComparisonModel] = Field(default_factory=list)
    summary: Dict[str, Any] = Field(default_factory=dict)
    consolidated_data: Dict[str, Any] = Field(default_factory=dict)  # Added for compatibility

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
    Extract financial data from a single document
    
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
        # Process the file
        result = await process_single_file(temp_file_path, file.filename, openai_api_key)
        return result
        
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

@app.post("/extract-multiple", response_model=MultiFileResultModel)
async def extract_multiple(
    files: List[UploadFile] = File(...),
    api_key: str = Depends(get_api_key),
    openai_api_key: str = Depends(get_openai_api_key),
    generate_comparisons: bool = Form(True)
):
    """
    Extract financial data from multiple documents and generate comparisons
    
    Args:
        files: List of uploaded documents (PDF, Excel, CSV, TXT)
        api_key: API key for authentication
        openai_api_key: OpenAI API key
        generate_comparisons: Whether to generate comparisons between documents
        
    Returns:
        Extracted financial data and comparisons
    """
    logger.info(f"Processing {len(files)} files")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Create temp directory for files
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    try:
        # Save all files to temp directory
        for file in files:
            # Check file extension
            file_ext = os.path.splitext(file.filename)[1].lower().lstrip('.')
            if file_ext not in ['pdf', 'xlsx', 'xls', 'csv', 'txt']:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
            
            # Save file
            temp_file_path = os.path.join(temp_dir, f"{file.filename}")
            with open(temp_file_path, "wb") as f:
                f.write(await file.read())
            
            temp_files.append((temp_file_path, file.filename))
        
        # Process all files
        results = []
        for temp_file_path, filename in temp_files:
            try:
                result = await process_single_file(temp_file_path, filename, openai_api_key)
                results.append(result)
                logger.info(f"Successfully processed {filename}")
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                # Continue processing other files even if one fails
        
        # Generate comparisons if requested and if we have multiple files
        comparisons = []
        if generate_comparisons and len(results) > 1:
            comparisons = generate_document_comparisons(results)
        
        # Generate summary
        summary = generate_summary(results)
        
        # Create consolidated data in the format expected by the NOI tool
        consolidated_data = create_consolidated_data(results)
        
        return {
            "documents": results,
            "comparisons": comparisons,
            "summary": summary,
            "consolidated_data": consolidated_data
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
    
    finally:
        # Clean up temp files
        for temp_file_path, _ in temp_files:
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        # Remove temp directory
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)

async def process_single_file(file_path: str, filename: str, openai_api_key: str) -> ExtractedDataModel:
    """
    Process a single file and extract financial data
    
    Args:
        file_path: Path to the file
        filename: Original filename
        openai_api_key: OpenAI API key
        
    Returns:
        Extracted financial data
    """
    logger.info(f"Processing file: {filename}")
    
    # Step 1: Preprocess the file
    logger.info("Preprocessing file")
    try:
        preprocessed_data = preprocess_file(file_path)
        logger.info("Preprocessing completed successfully")
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error preprocessing file: {str(e)}")
    
    # Step 2: Classify the document
    logger.info("Classifying document")
    file_ext = os.path.splitext(filename)[1].lower().lstrip('.')
    combined_text = preprocessed_data.get('content', {}).get('combined_text', '')
    if not combined_text:
        # Try to get text from different sources based on file type
        if file_ext in ['pdf', 'txt']:
            combined_text = '\n'.join([page.get('content', '') for page in preprocessed_data.get('content', {}).get('text', [])])
        elif file_ext in ['xlsx', 'xls', 'csv']:
            combined_text = preprocessed_data.get('content', {}).get('text_representation', '')
    
    try:
        classification_result = classify_document(combined_text, openai_api_key, filename)
        logger.info(f"Classification result: {classification_result}")
        
        # Ensure document_type and period are valid strings
        document_type = classification_result.get('document_type', 'Unknown')
        if document_type is None:
            document_type = 'Unknown'
            
        # Extract period from filename if not found by classifier
        period = classification_result.get('period', '')
        if not period:
            # Try to extract from filename (e.g., Actual_Mar_2025.xlsx)
            filename_parts = os.path.splitext(filename)[0].split('_')
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
                    year_match = re.search(year_pattern, part)
                    if year_match:
                        year = year_match.group(0)
                
                if month and year:
                    period = f"{month} {year}"
                elif year:
                    period = year
            
            logger.info(f"Extracted period from filename: {period}")
        
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

def create_consolidated_data(documents: List[ExtractedDataModel]) -> Dict[str, Any]:
    """
    Create consolidated data in the format expected by the NOI tool
    
    Args:
        documents: List of extracted document data
        
    Returns:
        Consolidated data dictionary
    """
    logger.info("Creating consolidated data for NOI tool compatibility")
    
    # Create a dictionary to store data by document type and period
    consolidated = {}
    
    for doc in documents:
        # Determine document category (actual, budget, prior_year)
        category = "unknown"
        if "actual" in doc.document_type.lower() and "prior" not in doc.document_type.lower():
            # Check if it's current year or prior year
            if "2024" in doc.period:
                category = "prior_year"
            else:
                category = "actual"
        elif "budget" in doc.document_type.lower():
            category = "budget"
        elif "prior" in doc.document_type.lower():
            category = "prior_year"
        
        # Extract month and year from period
        month = None
        year = None
        
        # Try to extract month
        month_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', doc.period, re.IGNORECASE)
        if month_match:
            month = month_match.group(0)
        
        # Try to extract year
        year_match = re.search(r'20\d{2}', doc.period)
        if year_match:
            year = year_match.group(0)
        
        # Create period key
        period_key = f"{month}_{year}" if month and year else doc.period.replace(" ", "_")
        
        # Create entry if it doesn't exist
        if period_key not in consolidated:
            consolidated[period_key] = {}
        
        # Add data for this category
        consolidated[period_key][category] = {
            "document_type": doc.document_type,
            "period": doc.period,
            "financials": doc.financials.__dict__
        }
    
    # Add metadata
    result = {
        "data": consolidated,
        "metadata": {
            "document_count": len(documents),
            "periods": list(consolidated.keys()),
            "has_budget": any("budget" in doc.document_type.lower() for doc in documents),
            "has_prior_year": any("prior" in doc.document_type.lower() or "2024" in doc.period for doc in documents)
        }
    }
    
    return result

def generate_document_comparisons(documents: List[ExtractedDataModel]) -> List[ComparisonModel]:
    """
    Generate comparisons between documents
    
    Args:
        documents: List of extracted document data
        
    Returns:
        List of comparisons
    """
    logger.info("Generating document comparisons")
    comparisons = []
    
    # Group documents by type and period
    actuals = {}
    budgets = {}
    prior_year_actuals = {}
    
    for doc in documents:
        if "actual" in doc.document_type.lower() and "prior" not in doc.document_type.lower():
            # Check if it's current year or prior year
            if "2024" in doc.period:
                prior_year_actuals[doc.period] = doc
            else:
                actuals[doc.period] = doc
        elif "budget" in doc.document_type.lower():
            budgets[doc.period] = doc
        elif "prior" in doc.document_type.lower():
            prior_year_actuals[doc.period] = doc
    
    # Generate Budget vs Actual comparisons
    for period in actuals.keys():
        if period in budgets:
            comparison = compare_documents(
                actuals[period], 
                budgets[period], 
                "budget_vs_actual", 
                period
            )
            comparisons.append(comparison)
    
    # Generate Year-over-Year comparisons
    for period in actuals.keys():
        # Extract year from period
        year_match = re.search(r'20\d{2}', period)
        if not year_match:
            continue
            
        current_year = year_match.group(0)
        prev_year = str(int(current_year) - 1)
        
        # Find matching period from previous year
        for prior_period in prior_year_actuals.keys():
            if prev_year in prior_period:
                comparison = compare_documents(
                    actuals[period], 
                    prior_year_actuals[prior_period], 
                    "year_over_year", 
                    period,
                    prior_period
                )
                comparisons.append(comparison)
    
    # Generate Month-over-Month comparisons
    periods_with_months = []
    for period in actuals.keys():
        # Extract month and year
        month_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)', period, re.IGNORECASE)
        year_match = re.search(r'20\d{2}', period)
        
        if month_match and year_match:
            month = month_match.group(0)
            year = year_match.group(0)
            periods_with_months.append((period, month, year))
    
    # Sort by year and month
    month_order = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    
    periods_with_months.sort(
        key=lambda x: (int(x[2]), month_order.get(x[1].lower()[:3], 0))
    )
    
    # Generate comparisons for adjacent months
    for i in range(1, len(periods_with_months)):
        current_period = periods_with_months[i][0]
        prev_period = periods_with_months[i-1][0]
        
        comparison = compare_documents(
            actuals[current_period], 
            actuals[prev_period], 
            "month_over_month", 
            current_period,
            prev_period
        )
        comparisons.append(comparison)
    
    return comparisons

def compare_documents(
    base_doc: ExtractedDataModel, 
    comparison_doc: ExtractedDataModel, 
    comparison_type: str,
    period: str,
    comparison_period: Optional[str] = None
) -> ComparisonModel:
    """
    Compare two documents and calculate variances
    
    Args:
        base_doc: Base document for comparison
        comparison_doc: Document to compare against
        comparison_type: Type of comparison
        period: Period of the base document
        comparison_period: Period of the comparison document
        
    Returns:
        Comparison results
    """
    variances = {}
    variance_percentages = {}
    
    # Calculate variances for all financial fields
    for field in base_doc.financials.__dict__.keys():
        base_value = getattr(base_doc.financials, field)
        comparison_value = getattr(comparison_doc.financials, field)
        
        if base_value is not None and comparison_value is not None:
            variance = base_value - comparison_value
            variances[field] = variance
            
            # Calculate percentage variance
            if comparison_value != 0:
                variance_percentage = (variance / abs(comparison_value)) * 100
                variance_percentages[field] = variance_percentage
            else:
                variance_percentages[field] = None
        else:
            variances[field] = None
            variance_percentages[field] = None
    
    return ComparisonModel(
        type=comparison_type,
        period=period,
        comparison_period=comparison_period,
        base_document=base_doc.document_type,
        comparison_document=comparison_doc.document_type,
        variances=variances,
        variance_percentages=variance_percentages
    )

def generate_summary(documents: List[ExtractedDataModel]) -> Dict[str, Any]:
    """
    Generate summary statistics from all documents
    
    Args:
        documents: List of extracted document data
        
    Returns:
        Summary statistics
    """
    logger.info("Generating summary statistics")
    
    summary = {
        "document_count": len(documents),
        "document_types": {},
        "periods": {},
        "average_noi": None,
        "total_revenue_range": {"min": None, "max": None},
        "total_expenses_range": {"min": None, "max": None},
        "noi_range": {"min": None, "max": None}
    }
    
    # Count document types
    for doc in documents:
        # Count document types
        doc_type = doc.document_type
        if doc_type in summary["document_types"]:
            summary["document_types"][doc_type] += 1
        else:
            summary["document_types"][doc_type] = 1
        
        # Count periods
        period = doc.period
        if period in summary["periods"]:
            summary["periods"][period] += 1
        else:
            summary["periods"][period] = 1
        
        # Calculate ranges
        financials = doc.financials
        
        # Total Revenue range
        if financials.total_revenue is not None:
            if summary["total_revenue_range"]["min"] is None or financials.total_revenue < summary["total_revenue_range"]["min"]:
                summary["total_revenue_range"]["min"] = financials.total_revenue
            if summary["total_revenue_range"]["max"] is None or financials.total_revenue > summary["total_revenue_range"]["max"]:
                summary["total_revenue_range"]["max"] = financials.total_revenue
        
        # Total Expenses range
        if financials.total_expenses is not None:
            if summary["total_expenses_range"]["min"] is None or financials.total_expenses < summary["total_expenses_range"]["min"]:
                summary["total_expenses_range"]["min"] = financials.total_expenses
            if summary["total_expenses_range"]["max"] is None or financials.total_expenses > summary["total_expenses_range"]["max"]:
                summary["total_expenses_range"]["max"] = financials.total_expenses
        
        # NOI range
        if financials.net_operating_income is not None:
            if summary["noi_range"]["min"] is None or financials.net_operating_income < summary["noi_range"]["min"]:
                summary["noi_range"]["min"] = financials.net_operating_income
            if summary["noi_range"]["max"] is None or financials.net_operating_income > summary["noi_range"]["max"]:
                summary["noi_range"]["max"] = financials.net_operating_income
    
    # Calculate average NOI
    noi_values = [doc.financials.net_operating_income for doc in documents if doc.financials.net_operating_income is not None]
    if noi_values:
        summary["average_noi"] = sum(noi_values) / len(noi_values)
    
    return summary

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Run the server
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=True)
