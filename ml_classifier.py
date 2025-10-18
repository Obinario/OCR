"""
ML Classifier module for report card classification and status verification.
This module handles all machine learning operations for document classification.
"""

import joblib
from typing import List, Dict, Optional, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLClassifier:
    """
    Machine Learning classifier for report card classification and status verification.
    
    This class handles:
    - Loading and managing ML models
    - Text classification (Report Card vs Not Report Card)
    - Report card status verification (pass/fail analysis)
    """
    
    def __init__(self, model_path: str = None, vectorizer_path: str = None):
        """
        Initialize the ML classifier with model and vectorizer.
        
        Args:
            model_path: Path to the trained model file
            vectorizer_path: Path to the vectorizer file
        """
        self.clf = None
        self.vectorizer = None
        self.model_loaded = False
        
        # Try to load models with fallback paths
        self._load_models(model_path, vectorizer_path)
    
    def _load_models(self, model_path: str = None, vectorizer_path: str = None) -> None:
        """
        Load ML model and vectorizer with fallback paths.
        
        Args:
            model_path: Custom model path
            vectorizer_path: Custom vectorizer path
        """
        try:
            # Try original models first (better accuracy)
            if not model_path:
                self.clf = joblib.load('report_card_model.pkl')
                self.vectorizer = joblib.load('vectorizer.pkl')
                logger.info("Original ML model and vectorizer loaded successfully!")
                self.model_loaded = True
                return
        except Exception as e:
            logger.warning(f"Failed to load original models: {e}")
        
        try:
            # Fallback to automated training models
            if not model_path:
                self.clf = joblib.load('models/auto_report_card_model.pkl')
                self.vectorizer = joblib.load('models/auto_vectorizer.pkl')
                logger.info("Automated ML model and vectorizer loaded successfully!")
                self.model_loaded = True
                return
        except Exception as e:
            logger.warning(f"Failed to load automated models: {e}")
        
        try:
            # Try custom paths if provided
            if model_path and vectorizer_path:
                self.clf = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                logger.info("Custom ML model and vectorizer loaded successfully!")
                self.model_loaded = True
                return
        except Exception as e:
            logger.error(f"Failed to load custom models: {e}")
        
        # If all attempts fail
        logger.error("Error loading ML model")
        logger.info("Please run 'python auto_train.py' to train the model automatically.")
        self.model_loaded = False
    
    def is_model_available(self) -> bool:
        """
        Check if the ML model is available and loaded.
        
        Returns:
            bool: True if model is loaded, False otherwise
        """
        return self.model_loaded and self.clf is not None and self.vectorizer is not None
    
    def classify_text(self, texts: List[Dict[str, Any]]) -> str:
        """
        Classify OCR text as Report Card or Not Report Card using ML model.
        
        Args:
            texts: List of dictionaries containing 'text' and 'confidence' keys
            
        Returns:
            str: Classification result ("Report Card", "Not Report Card", or error message)
        """
        if not self.is_model_available():
            return "Model not available"
        
        if not texts:
            return "No text to classify"
        
        try:
            # Merge all extracted text into a single string
            raw_text = " ".join([t['text'] for t in texts])
            
            # Transform text using the trained vectorizer
            X = self.vectorizer.transform([raw_text])
            
            # Make prediction
            prediction = self.clf.predict(X)[0]
            
            # Get prediction confidence
            confidence_scores = self.clf.predict_proba(X)[0]
            confidence = max(confidence_scores) * 100
            
            # Return result
            result = "Report Card" if prediction == 1 else "Not Report Card"
            logger.info(f"Classification: {result} (Confidence: {confidence:.1f}%)")
            
            return result
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            return "Classification error"
    
    def verify_report_card_status(self, texts: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Verify if a report card has failed or passed remarks.
        
        Args:
            texts: List of dictionaries containing 'text' and 'confidence' keys
            
        Returns:
            Dict[str, str]: Dictionary with 'status' and 'message' keys
        """
        if not texts:
            return {"status": "unknown", "message": "No text to analyze"}
        
        try:
            # Merge all extracted text into a single string
            raw_text = " ".join([t['text'] for t in texts])
            text_lower = raw_text.lower()
            
            # Primary failure remarks - these are the most reliable indicators of failed status
            # Look for specific patterns that indicate actual student failures, not grading scale info
            primary_failure_remarks = [
                'failed remarks', 'failed grade', 'failed subject', 'failed course',
                'failed in', 'has failed', 'student failed', 'grade failed',
                'failed mark', 'failed status', 'academic failure'
            ]
            
            # Exclude grading scale patterns that contain "failed" but aren't actual student failures
            grading_scale_patterns = [
                'below 75', 'did not meet expectations', 'grading scale', 'descriptors',
                'outstanding', 'very satisfactory', 'satisfactory', 'fairly satisfactory'
            ]
            
            # Secondary failure indicators - additional context clues
            secondary_failure_keywords = [
                'incomplete', 'incomplete grade', 'needs improvement', 
                'unsatisfactory', 'remedial', 'retake', 'repeat', 
                'not passed', 'did not pass', 'conditional', 'probation', 
                'academic warning', 'insufficient', 'deficient', 'below average'
            ]
            
            # Keywords that indicate passing
            passing_keywords = [
                'passed', 'pass', 'satisfactory', 'good', 'excellent',
                'outstanding', 'very good', 'above average', 'promoted',
                'promotion', 'completed', 'successful', 'achieved'
            ]
            
            # Check for primary failure remarks (most reliable)
            primary_failure_found = False
            found_primary_remarks = []
            
            # First, check if we're in a grading scale context (which should be ignored)
            is_grading_scale_context = any(pattern in text_lower for pattern in grading_scale_patterns)
            
            for remark in primary_failure_remarks:
                if remark in text_lower:
                    # Only consider it a failure if it's not in grading scale context
                    if not is_grading_scale_context:
                        primary_failure_found = True
                        found_primary_remarks.append(remark)
                        logger.info(f"Found primary failure remark: '{remark}'")
                    else:
                        logger.info(f"Ignoring '{remark}' as it appears to be in grading scale context")
            
            # Check for actual subject failures in the grade tables
            # Look for patterns like "Subject Name" followed by grades and "Failed" remarks
            subject_failure_patterns = [
                'failed', 'fail', 'incomplete', 'unsatisfactory'
            ]
            
            # Look for actual subject failures by checking if "failed" appears in subject context
            # This helps distinguish between grading scale "Failed" and actual subject "Failed"
            actual_subject_failures = 0
            if 'failed' in text_lower and not is_grading_scale_context:
                # Look for patterns that suggest actual subject failures
                # Check if "failed" appears near subject names or grade numbers
                import re
                # Look for "failed" that's not part of grading scale explanations
                failed_matches = re.finditer(r'\bfailed\b', text_lower)
                for match in failed_matches:
                    start = max(0, match.start() - 50)
                    end = min(len(text_lower), match.end() + 50)
                    context = text_lower[start:end]
                    
                    # Check if this "failed" is in subject/grade context, not grading scale
                    if any(subject_word in context for subject_word in ['subject', 'grade', 'quarter', 'semester', 'final']):
                        if not any(scale_word in context for scale_word in ['descriptors', 'grading scale', 'below 75', 'outstanding', 'satisfactory']):
                            actual_subject_failures += 1
                            logger.info(f"Found actual subject failure in context: {context}")
            
            # If we found actual subject failures, treat as primary failure
            if actual_subject_failures > 0:
                primary_failure_found = True
                found_primary_remarks.append(f"Actual subject failures detected ({actual_subject_failures})")
                logger.info(f"Found {actual_subject_failures} actual subject failures")
            
            # Check for secondary failure indicators
            secondary_failure_count = 0
            found_secondary_keywords = []
            for keyword in secondary_failure_keywords:
                if keyword in text_lower:
                    # Only count if not in grading scale context
                    if not is_grading_scale_context:
                        secondary_failure_count += 1
                        found_secondary_keywords.append(keyword)
                        logger.info(f"Found secondary failure keyword: '{keyword}'")
                    else:
                        logger.info(f"Ignoring '{keyword}' as it appears to be in grading scale context")
            
            # Check for passing indicators
            passing_count = 0
            found_passing_keywords = []
            for keyword in passing_keywords:
                if keyword in text_lower:
                    passing_count += 1
                    found_passing_keywords.append(keyword)
                    logger.info(f"Found passing keyword: '{keyword}'")
            
            # Determine status based on analysis
            if primary_failure_found:
                # Primary failure remarks found - definitely failed
                status = "failed"
                message = "You have failed remarks"
                if found_primary_remarks:
                    message += f" (Found: {', '.join(found_primary_remarks)})"
            elif secondary_failure_count > 0 and not passing_count:
                # Only secondary failure keywords and no passing indicators
                status = "failed"
                message = "You have failed remarks"
                if found_secondary_keywords:
                    message += f" (Found: {', '.join(found_secondary_keywords)})"
            elif passing_count > 0 and not primary_failure_found and secondary_failure_count == 0:
                # Only passing keywords and no failure indicators
                status = "passed"
                message = "You have passed"
            elif passing_count > 0 and (primary_failure_found or secondary_failure_count > 0):
                # Mixed indicators - failure takes precedence
                status = "failed"
                message = "You have failed remarks"
                if found_primary_remarks:
                    message += f" (Found: {', '.join(found_primary_remarks)})"
                elif found_secondary_keywords:
                    message += f" (Found: {', '.join(found_secondary_keywords)})"
            else:
                # No clear indicators found
                status = "unknown"
                message = "Unable to determine pass/fail status"
            
            logger.info(f"Report card status analysis: {status} - {message}")
            return {"status": status, "message": message}
            
        except Exception as e:
            logger.error(f"Error in report card status verification: {str(e)}")
            return {"status": "error", "message": f"Error analyzing report card: {str(e)}"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dict[str, Any]: Model information including availability and type
        """
        return {
            "model_loaded": self.model_loaded,
            "classifier_available": self.clf is not None,
            "vectorizer_available": self.vectorizer is not None,
            "model_type": type(self.clf).__name__ if self.clf else None,
            "vectorizer_type": type(self.vectorizer).__name__ if self.vectorizer else None
        }
