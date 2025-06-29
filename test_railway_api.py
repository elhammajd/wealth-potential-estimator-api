#!/usr/bin/env python3
"""
Test the deployed Railway API with local test images
"""

import requests
import json
import os
from pathlib import Path

API_URL = "https://wealth-potential-estimator-api-production.up.railway.app"

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Health check: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_predict_with_image(image_path):
    """Test the predict endpoint with an image"""
    try:
        with open(image_path, 'rb') as image_file:
            files = {'file': (os.path.basename(image_path), image_file, 'image/jpeg')}
            response = requests.post(f"{API_URL}/predict", files=files, timeout=30)
            
        print(f"\n--- Testing with {os.path.basename(image_path)} ---")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Estimated Net Worth: ${result['estimated_net_worth']:,.2f}")
            print("Top 3 Matches:")
            for i, match in enumerate(result['top_matches'], 1):
                print(f"  {i}. {match['name']} (similarity: {match['similarity_score']:.6f})")
                print(f"      Net Worth: ${match['net_worth']:,.2f}")
                print(f"      Age: {match['age']}, Source: {match['source']}")
        else:
            print(f"Error: {response.text}")
            
        return response.status_code == 200
    except Exception as e:
        print(f"Error testing {image_path}: {e}")
        return False

def test_all_images():
    """Test all images in test_images folder"""
    print("=== Testing Railway Deployed API ===")
    print(f"API URL: {API_URL}")
    
    # Test health endpoint
    if not test_health():
        print("Health check failed. API might not be ready.")
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
    
    print(f"\n=== Railway API Test Results ===")
    print(f"Successfully tested {success_count}/{len(image_files)} images")
    print(f"API URL: {API_URL}")

if __name__ == "__main__":
    test_all_images()
