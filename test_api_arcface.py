#!/usr/bin/env python3
"""
Test the API with ArcFace implementation
"""

import requests
import json

def test_api_with_arcface():
    """Test with a single image to see the response format"""
    try:
        with open("test_images/business_person.jpg", 'rb') as image_file:
            files = {'file': ('business_person.jpg', image_file, 'image/jpeg')}
            response = requests.post("http://localhost:8000/predict", files=files)
            
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Estimated Net Worth: ${result['estimated_net_worth']:,.2f}")
            print("Top 3 Matches:")
            for i, match in enumerate(result['top_matches'], 1):
                print(f"  {i}. {match['name']} (similarity: {match['similarity_score']:.6f})")
                print(f"      Net Worth: ${match['net_worth']:,.2f}")
        else:
            print(f"Error response: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api_with_arcface()
