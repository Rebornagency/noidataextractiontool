"""
Fixed Preprocessing Module and Document Classifier Integration
This module fixes the type mismatch between preprocessing_module and document_classifier
"""

import os
import logging
import json
import re
from typing import Dict, Any, List, Tuple, Optional, Union
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('document_classifier')

class DocumentClassifier:
    """
    Class for classifying financial documents and extracting time periods
    using GPT-4
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the document classifier
        
        Args:
            api_key: OpenAI API key (optional, can be set via environment variable)
        """
        # Set API key if provided, otherwise use environment variable
        if api_key:
            openai.api_key = api_key
        else:
            openai.api_key = os.environ.get("OPENAI_API_KEY")
            
        if not openai.api_key:
            logger.warning("OpenAI API key not set. Please set OPENAI_API_KEY environment variable or provide it during initialization.")
        
        # Document types we can classify
        self.document_types = [
            "Actual Income Statement",
            "Budget",
            "Prior Year Actual",
            "Unknown"
        ]
    
    def classify(self, text_or_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Classify document type and extract time period
        
        Args:
            text_or_data: Preprocessed text from the document or data dictionary
            
        Returns:
            Dict containing document type and period
        """
        logger.info("Classifying document and extracting time period")
        
        # Handle both string and dictionary input
        if isinstance(text_or_data, dict):
            # Extract text from dictionary if it's a dictionary
            logger.info("Input is a dictionary, extracting text content")
            
            # Try to extract text from common dictionary structures
            if 'text' in text_or_data:
                text = text_or_data['text']
            elif 'content' in text_or_data:
                text = text_or_data['content']
            elif 'data' in text_or_data and isinstance(text_or_data['data'], str):
                text = text_or_data['data']
            else:
                # Convert the entire dictionary to a string as fallback
                text = json.dumps(text_or_data)
                logger.warning("Could not find text field in dictionary, using JSON string representation")
        else:
            # Use the input directly if it's already a string
            text = text_or_data
        
        # Ensure text is a string
        if not isinstance(text, str):
            text = str(text)
            logger.warning(f"Converted non-string input to string: {type(text_or_data)}")
        
        # First try rule-based classification for efficiency
        rule_based_result = self._rule_based_classification(text)
        
        # If rule-based classification is confident, return the result
        if rule_based_result.get('confidence', 0) > 0.8:
            logger.info(f"Rule-based classification successful: {rule_based_result}")
            return {
                'document_type': rule_based_result['document_type'],
                'period': rule_based_result['period'],
                'method': 'rule_based'
            }
        
        # Otherwise, use GPT for classification
        gpt_result = self._gpt_classification(text)
        logger.info(f"GPT classification result: {gpt_result}")
        
        return {
            'document_type': gpt_result['document_type'],
            'period': gpt_result['period'],
            'method': 'gpt'
        }
    
    def _rule_based_classification(self, text: str) -> Dict[str, Any]:
        """
        Attempt to classify document using rule-based approach
        
        Args:
            text: Preprocessed text from the document
            
        Returns:
            Dict containing document type, period, and confidence
        """
        result = {
            'document_type': 'Unknown',
            'period': None,
            'confidence': 0.0
        }
        
        # Convert to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Check for document type indicators
        if 'actual' in text_lower and ('income statement' in text_lower or 'statement of income' in text_lower):
            result['document_type'] = 'Actual Income Statement'
            result['confidence'] = 0.7
        elif 'budget' in text_lower:
            result['document_type'] = 'Budget'
            result['confidence'] = 0.7
        elif 'prior year' in text_lower or 'previous year' in text_lower:
            result['document_type'] = 'Prior Year Actual'
            result['confidence'] = 0.7
            
        # Extract period using regex patterns
        # Pattern for month and year (e.g., "March 2025", "Jan 2025", "January 2025")
        month_year_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+\d{4}'
        
        # Pattern for quarter (e.g., "Q1 2025", "First Quarter 2025")
        quarter_pattern = r'(?:Q[1-4]|(?:First|Second|Third|Fourth)\s+Quarter)[,\s]+\d{4}'
        
        # Pattern for year only (e.g., "2025", "FY 2025")
        year_pattern = r'(?:FY\s+)?\d{4}'
        
        # Try to find month and year
        month_year_match = re.search(month_year_pattern, text, re.IGNORECASE)
        if month_year_match:
            result['period'] = month_year_match.group(0).strip()
            result['confidence'] += 0.2
            return result
            
        # Try to find quarter
        quarter_match = re.search(quarter_pattern, text, re.IGNORECASE)
        if quarter_match:
            result['period'] = quarter_match.group(0).strip()
            result['confidence'] += 0.2
            return result
            
        # Try to find year only
        year_match = re.search(year_pattern, text)
        if year_match:
            result['period'] = year_match.group(0).strip()
            result['confidence'] += 0.1
            return result
            
        return result
    
    def _gpt_classification(self, text: str) -> Dict[str, Any]:
        """
        Classify document using GPT-4
        
        Args:
            text: Preprocessed text from the document
            
        Returns:
            Dict containing document type and period
        """
        # Prepare a sample of the text for GPT (first 1000 characters)
        text_sample = text[:1000]
        
        # Create the prompt for GPT
        prompt = f"""Classify this document as one of:
- Actual Income Statement
- Budget
- Prior Year Actual
- Unknown

Then extract the month and year.

Text:
"""
        prompt += text_sample
        
        try:
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a senior real estate accountant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more deterministic results
                max_tokens=100
            )
            
            # Extract response text
            response_text = response.choices[0].message.content.strip()
            
            # Parse the response
            document_type = 'Unknown'
            period = None
            
            # Check for document type
            for doc_type in self.document_types:
                if doc_type in response_text:
                    document_type = doc_type
                    break
            
            # Extract period using regex patterns
            month_year_pattern = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+\d{4}'
            quarter_pattern = r'(?:Q[1-4]|(?:First|Second|Third|Fourth)\s+Quarter)[,\s]+\d{4}'
            year_pattern = r'(?:FY\s+)?\d{4}'
            
            month_year_match = re.search(month_year_pattern, response_text, re.IGNORECASE)
            if month_year_match:
                period = month_year_match.group(0).strip()
            else:
                quarter_match = re.search(quarter_pattern, response_text, re.IGNORECASE)
                if quarter_match:
                    period = quarter_match.group(0).strip()
                else:
                    year_match = re.search(year_pattern, response_text)
                    if year_match:
                        period = year_match.group(0).strip()
            
            return {
                'document_type': document_type,
                'period': period
            }
            
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            # Fallback to Unknown if API call fails
            return {
                'document_type': 'Unknown',
                'period': None
            }


def classify_document(text_or_data: Union[str, Dict[str, Any]], api_key: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """
    Convenience function to classify a document
    
    Args:
        text_or_data: Preprocessed text from the document or data dictionary
        api_key: OpenAI API key (optional)
        
    Returns:
        Tuple of (document_type, period)
    """
    classifier = DocumentClassifier(api_key)
    
    try:
        # Log input type for debugging
        logger.info(f"classify_document input type: {type(text_or_data)}")
        
        result = classifier.classify(text_or_data)
        return result['document_type'], result['period']
    except Exception as e:
        logger.error(f"Error in classify_document: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "Unknown", None


def extract_period_from_filename(filename: str) -> str:
    """
    Extract period information from filename
    
    Args:
        filename: Original filename
        
    Returns:
        Extracted period or "Unknown Period" if not found
    """
    if not filename:
        return "Unknown Period"
            
    # Remove file extension
    filename_parts = os.path.splitext(filename)[0].split('_')
    
    # Define month abbreviations and pattern for year
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
    
    # Construct period string
    if month and year:
        return f"{month} {year}"
    elif year:
        return year
            
    return "Unknown Period"


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python document_classifier.py <text_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    try:
        with open(file_path, 'r') as f:
            text = f.read()
        
        doc_type, period = classify_document(text)
        result = {
            "document_type": doc_type,
            "period": period
        }
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
