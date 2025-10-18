#!/usr/bin/env python3
"""
Automated OCR Training System
Automatically collects OCR data and trains the classification model
"""

import os
import json
import cv2
import numpy as np
import joblib
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from paddleocr import PaddleOCR

class AutoTrainer:
    def __init__(self):
        """Initialize the automated trainer"""
        self.ocr = None
        self.data_file = "auto_training_data.json"
        self.model_file = "auto_report_card_model.pkl"
        self.vectorizer_file = "auto_vectorizer.pkl"
        self.setup_directories()
        self.initialize_ocr()
        
    def setup_directories(self):
        """Create necessary directories"""
        if not os.path.exists("training_images"):
            os.makedirs("training_images")
        if not os.path.exists("models"):
            os.makedirs("models")
            
    def initialize_ocr(self):
        """Initialize PaddleOCR"""
        try:
            self.ocr = PaddleOCR(lang='en', use_textline_orientation=False)
            print("PaddleOCR initialized successfully!")
        except Exception as e:
            print(f"Error initializing PaddleOCR: {e}")
            self.ocr = None
    
    def scan_image(self, image_path):
        """Scan image and extract OCR text"""
        if not self.ocr:
            return None
            
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            # Add padding and preprocess
            padded_image = self.add_padding(image)
            processed_image = self.preprocess_image(padded_image)
            
            # Extract text using OCR
            result = self.ocr.ocr(padded_image)
            
            if not result or not result[0] or len(result[0]) == 0:
                return None
                
            # Process OCR results
            extracted_texts = []
            if isinstance(result[0], dict):  # New format
                if 'rec_texts' in result[0] and 'rec_scores' in result[0]:
                    texts = result[0]['rec_texts']
                    scores = result[0]['rec_scores']
                    for i, (text, score) in enumerate(zip(texts, scores)):
                        extracted_texts.append({
                            'text': text,
                            'confidence': round(score * 100, 2)
                        })
            elif isinstance(result[0], list):  # Old format
                for line in result[0]:
                    if line and len(line) >= 2 and len(line[1]) >= 2:
                        text = line[1][0]
                        confidence = line[1][1]
                        extracted_texts.append({
                            'text': text,
                            'confidence': round(confidence * 100, 2)
                        })
            
            # Merge all text
            full_text = " ".join([item['text'] for item in extracted_texts])
            return full_text
            
        except Exception as e:
            print(f"❌ Error scanning image: {e}")
            return None
    
    def add_padding(self, image, top=40, bottom=150, left=40, right=40, color=(255, 255, 255)):
        """Add padding around the image"""
        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )
        return padded
    
    def preprocess_image(self, image):
        """Preprocess image for better OCR"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            binary = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            kernel = np.ones((1, 1), np.uint8)
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            processed_image = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR)
            return processed_image
            
        except Exception as e:
            return image
    
    def auto_classify_text(self, text):
        """Automatically classify text based on keywords"""
        text_lower = text.lower()
        
        # Report card keywords (higher weight)
        report_keywords = [
            'report card', 'progress report', 'student report', 'academic report',
            'grade', 'semester', 'quarter', 'student name', 'school year',
            'mathematics', 'english', 'science', 'social studies', 'filipino',
            'general average', 'final grade', 'passed', 'failed', 'conduct',
            'curriculum', 'strand', 'adviser', 'section', 'grade level'
        ]
        
        # Non-report card keywords
        non_report_keywords = [
            'restaurant', 'menu', 'invoice', 'bill', 'payment', 'total amount',
            'recipe', 'ingredients', 'bake', 'cook', 'business card', 'phone',
            'advertisement', 'sale', 'discount', 'prescription', 'medication',
            'news article', 'event', 'flyer', 'resume', 'experience', 'skills',
            'product manual', 'instructions', 'manual'
        ]
        
        # Count keyword matches
        report_score = sum(1 for keyword in report_keywords if keyword in text_lower)
        non_report_score = sum(1 for keyword in non_report_keywords if keyword in text_lower)
        
        # Auto-classify based on keyword scores
        if report_score > non_report_score and report_score > 2:
            return 1, "Report Card"
        elif non_report_score > report_score and non_report_score > 1:
            return 0, "Not Report Card"
        else:
            # If unclear, use length and structure heuristics
            if len(text.split()) > 50 and any(word in text_lower for word in ['grade', 'student', 'subject']):
                return 1, "Report Card"
            else:
                return 0, "Not Report Card"
    
    def collect_training_data(self, data_directory="training_images"):
        """Automatically collect training data from images and text files"""
        print("Collecting training data automatically...")
        
        if not os.path.exists(data_directory):
            print(f"Directory not found: {data_directory}")
            return []
        
        # Find all image and text files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}
        text_extensions = {'.txt'}
        
        image_files = []
        text_files = []
        
        for root, dirs, files in os.walk(data_directory):
            for file in files:
                file_path = os.path.join(root, file)
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(file_path)
                elif any(file.lower().endswith(ext) for ext in text_extensions):
                    text_files.append(file_path)
        
        total_files = len(image_files) + len(text_files)
        if total_files == 0:
            print(f"No image or text files found in {data_directory}")
            return []
        
        print(f"Found {len(image_files)} images and {len(text_files)} text files")
        
        training_data = []
        processed_count = 0
        
        # Process text files first (faster)
        for i, text_path in enumerate(text_files):
            print(f"Processing text file {i+1}/{len(text_files)}: {os.path.basename(text_path)}")
            
            try:
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
                
                if text and len(text) > 10:
                    # Auto-classify
                    label, label_text = self.auto_classify_text(text)
                    
                    training_data.append((text, label))
                    processed_count += 1
                    
                    print(f"Loaded text ({len(text)} chars) -> {label_text}")
                else:
                    print(f"Empty or too short text file")
            except Exception as e:
                print(f"Error reading text file: {e}")
        
        # Process image files
        for i, image_path in enumerate(image_files):
            print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
            
            # Scan image
            text = self.scan_image(image_path)
            
            if text and len(text.strip()) > 10:  # Only use if we got meaningful text
                # Auto-classify
                label, label_text = self.auto_classify_text(text)
                
                training_data.append((text, label))
                processed_count += 1
                
                print(f"Extracted text ({len(text)} chars) -> {label_text}")
            else:
                print(f"No meaningful text extracted")
        
        print(f"\nCollection complete: {processed_count}/{total_files} files processed")
        return training_data
    
    def add_hardcoded_examples(self):
        """Add some hardcoded examples for better initial training"""
        hardcoded_data = [
            # Report Card examples
            ("STUDENT REPORT CARD School Year 2023-2024 Student Name: John Doe Grade Level: 10 Section: A Academic Performance Mathematics 85 English 90 Science 88 Social Studies 92 Filipino 87 MAPEH 89 Computer 91 Average Grade: 89.0 Status: PASSED", 1),
            ("PROGRESS REPORT CARD Name: Maria Santos Grade: 11 Section: STEM-A Quarter: 2nd Subjects Grades Filipino 95 English 92 Mathematics 88 Science 90 Social Studies 93 PE 94 Computer 89 General Average: 91.6 Remarks: PASSED", 1),
            ("REPORT CARD Academic Year 2024-2025 Student: Juan Dela Cruz Grade: 9 Section: B Subjects Quarter 1 Quarter 2 Final Grade Mathematics 85 88 87 English 90 92 91 Science 88 90 89 Social Studies 92 94 93 Filipino 87 89 88 MAPEH 89 91 90 Computer 91 93 92 General Average: 90.0 Status: PASSED", 1),
            
            # Non-Report Card examples
            ("Welcome to our restaurant! Today's special menu includes: Grilled Chicken $12.99 Beef Steak $15.99 Fish Fillet $11.99 Vegetable Salad $8.99 Soup of the Day $6.99 Please call us at (555) 123-4567 for reservations", 0),
            ("INVOICE Invoice Number: INV-2024-001 Date: January 15, 2024 Bill To: ABC Company Address: 123 Main Street City: New York Items Description Quantity Price Total Office Supplies 10 $5.00 $50.00 Software License 1 $299.00 $299.00 Total Amount: $349.00 Payment Due: February 15, 2024", 0),
            ("RECIPE: Chocolate Chip Cookies Ingredients: 2 cups flour, 1 cup sugar, 1/2 cup butter, 2 eggs, 1 tsp vanilla, 1 cup chocolate chips Instructions: Mix dry ingredients. Cream butter and sugar. Add eggs and vanilla. Combine with dry ingredients. Fold in chocolate chips. Bake at 375°F for 10-12 minutes", 0),
        ]
        
        return hardcoded_data
    
    def train_model(self, training_data):
        """Train the classification model"""
        if len(training_data) < 5:
            print("Very few training examples. Adding hardcoded examples...")
            training_data.extend(self.add_hardcoded_examples())
        
        print(f"\nTraining model with {len(training_data)} examples...")
        
        # Separate text and labels
        texts = [item[0] for item in training_data]
        labels = [item[1] for item in training_data]
        
        # Count examples by class
        report_cards = sum(labels)
        not_report_cards = len(labels) - report_cards
        
        print(f"Training Data:")
        print(f"  Report Cards: {report_cards}")
        print(f"  Not Report Cards: {not_report_cards}")
        print(f"  Total: {len(training_data)}")
        
        # Split data for training and testing
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Initialize TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Transform text data
        print("Vectorizing text data...")
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Initialize and train Random Forest classifier
        print("Training Random Forest classifier...")
        clf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        clf.fit(X_train_tfidf, y_train)
        
        # Evaluate the model
        print("Evaluating model...")
        y_pred = clf.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Not Report Card', 'Report Card']))
        
        # Save the model and vectorizer
        print(f"\nSaving model...")
        model_path = os.path.join("models", self.model_file)
        vectorizer_path = os.path.join("models", self.vectorizer_file)
        
        joblib.dump(clf, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        
        print(f"Model saved to: {model_path}")
        print(f"Vectorizer saved to: {vectorizer_path}")
        
        return clf, vectorizer
    
    def test_model(self, clf, vectorizer):
        """Test the trained model"""
        print(f"\nTesting model...")
        
        test_samples = [
            "PROGRESS REPORT CARD Name: Test Student Grade: 10 Mathematics 85 English 90 Science 88 Average: 87.7 Status: PASSED",
            "RESTAURANT MENU Today's Special: Grilled Salmon $18.99 Chicken Alfredo $16.99 Vegetarian Pasta $14.99 Call (555) 123-4567",
            "STUDENT REPORT CARD Academic Year 2024-2025 Name: Maria Garcia Grade: 11 Subjects: Math 95 English 92 Science 88 Average: 91.7"
        ]
        
        print(f"\nTest Predictions:")
        for i, text in enumerate(test_samples):
            X = vectorizer.transform([text])
            prediction = clf.predict(X)[0]
            probability = clf.predict_proba(X)[0]
            
            result = "Report Card" if prediction == 1 else "Not Report Card"
            confidence = max(probability) * 100
            
            print(f"Test {i+1}: {result} (Confidence: {confidence:.1f}%)")
            print(f"  Text: {text[:60]}...")
    
    def auto_train(self, images_directory="training_images"):
        """Complete automated training process"""
        print("Automated OCR Training System")
        print("=" * 50)
        
        # Step 1: Collect training data
        training_data = self.collect_training_data(images_directory)
        
        if not training_data:
            print("No training data collected. Please add images to the training_images directory.")
            return False
        
        # Step 2: Train model
        clf, vectorizer = self.train_model(training_data)
        
        # Step 3: Test model
        self.test_model(clf, vectorizer)
        
        print(f"\nAutomated training complete!")
        print(f"Models saved in: models/")
        print(f"Update your Flask app to use the new models")
        
        return True

def main():
    """Main function"""
    trainer = AutoTrainer()
    
    print("Automated OCR Training System")
    print("=" * 50)
    print("This system will:")
    print("1. Scan images in the 'training_images' directory")
    print("2. Automatically classify them as Report Card or Not Report Card")
    print("3. Train a machine learning model")
    print("4. Save the trained model for use in your Flask app")
    print()
    
    # Check if training images directory exists
    if not os.path.exists("training_images"):
        print("Creating training_images directory...")
        os.makedirs("training_images")
        print("Please add your document images to the 'training_images' directory")
        print("   - Report cards, progress reports, etc.")
        print("   - Menus, invoices, articles, etc.")
        print("   - Then run this script again")
        return
    
    # Check if there are images or text files
    files_found = []
    for root, dirs, files in os.walk("training_images"):
        for file in files:
            if any(file.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.txt']):
                files_found.append(os.path.join(root, file))
    
    if not files_found:
        print("No images or text files found in training_images directory")
        print("Please add document images or text files to the 'training_images' directory")
        return
    
    print(f"Found {len(files_found)} files in training_images directory")
    
    # Start automated training
    success = trainer.auto_train()
    
    if success:
        print(f"\nNext steps:")
        print(f"1. Update app.py to load models from 'models/' directory")
        print(f"2. Test your Flask app with the new model")
        print(f"3. Add more images to training_images/ for better accuracy")

if __name__ == "__main__":
    main()
