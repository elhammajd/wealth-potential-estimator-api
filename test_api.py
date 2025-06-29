#!/usr/bin/env python3
"""
Test script for the Wealth Potential Estimator API
"""

import requests
import os
import json
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_predict_with_image(image_path):
    """Test the predict endpoint with an image"""
    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}
            response = requests.post(f"{API_URL}/predict", files=files)
            
        print(f"\n--- Testing with {os.path.basename(image_path)} ---")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Estimated Net Worth: ${result['estimated_net_worth']:,.2f}")
            print("Top 3 Matches:")
            for i, match in enumerate(result['top_matches'], 1):
                print(f"  {i}. {match['name']} (similarity: {match['similarity_score']:.3f})")
                print(f"      Net Worth: ${match['net_worth']:,.2f}")
                print(f"      Age: {match['age']}, Source: {match['source']}")
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing {image_path}: {e}")
        return False

def main():
    """Main test function"""
    print("=== Wealth Potential Estimator API Test ===\n")
    
    # Test health endpoint
    if not test_health():
        print("Health check failed. Make sure the server is running.")
        return
    
    # Get list of test images
    test_images_dir = Path("test_images")
    if not test_images_dir.exists():
        print("test_images directory not found!")
        return
    
    # Test with all images in test_images folder
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(test_images_dir.glob(ext))
    
    if not image_files:
        print("No image files found in test_images directory!")
        return
    
    print(f"\nFound {len(image_files)} images to test:")
    for img in image_files:
        print(f"  - {img.name}")
    
    # Test each image
    success_count = 0
    for image_path in image_files:
        if test_predict_with_image(image_path):
            success_count += 1
    
    print(f"\n=== Test Results ===")
    print(f"Successfully tested {success_count}/{len(image_files)} images")

if __name__ == "__main__":
    main()
