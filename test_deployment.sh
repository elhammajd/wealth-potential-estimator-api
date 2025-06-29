#!/bin/bash

# Quick test script for the deployed API
# Usage: ./test_deployment.sh https://your-api-url.up.railway.app

if [ -z "$1" ]; then
    echo "Need a URL to test!"
    echo "Usage: ./test_deployment.sh https://your-api-url.up.railway.app"
    exit 1
fi

API_URL=$1
echo "Testing API at: $API_URL"
echo "=========================================="

# Test 1: Health Check
echo "1. Checking if it's alive..."
HEALTH_RESPONSE=$(curl -s "$API_URL/health")
if echo "$HEALTH_RESPONSE" | grep -q "healthy"; then
    echo "Health check passed!"
    echo "   Response: $HEALTH_RESPONSE"
else
    echo "Health check failed!"
    echo "   Response: $HEALTH_RESPONSE"
    exit 1
fi

echo ""

# Test 2: API Documentation
echo "2. Checking docs page..."
DOCS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/docs")
if [ "$DOCS_STATUS" = "200" ]; then
    echo "Docs are working: $API_URL/docs"
else
    echo "Docs might not be working (status: $DOCS_STATUS)"
fi

echo ""

# Test 3: Prediction endpoint
echo "3. Testing the main prediction endpoint..."

# Try to find a test image
TEST_IMAGE=""
for img in "test_images/business_person.jpg" "test_images/download (1).jpeg" "test_images/casual_person.jpg"; do
    if [ -f "$img" ]; then
        TEST_IMAGE="$img"
        break
    fi
done

if [ -n "$TEST_IMAGE" ]; then
    echo "Testing with: $TEST_IMAGE"
    RESPONSE=$(curl -s -X POST "$API_URL/predict" \
        -H "Content-Type: multipart/form-data" \
        -F "file=@$TEST_IMAGE")
    
    if echo "$RESPONSE" | grep -q "estimated_net_worth"; then
        echo "Prediction working!"
        
        # Parse and display the response nicely
        NET_WORTH=$(echo "$RESPONSE" | grep -o '"estimated_net_worth":[0-9.]*' | cut -d':' -f2)
        if [ -n "$NET_WORTH" ]; then
            # Convert to millions/billions for readability
            if [ $(echo "$NET_WORTH > 1000000000" | bc -l) -eq 1 ]; then
                FORMATTED=$(echo "scale=1; $NET_WORTH / 1000000000" | bc -l)
                echo "   Estimated net worth: \$${FORMATTED}B"
            elif [ $(echo "$NET_WORTH > 1000000" | bc -l) -eq 1 ]; then
                FORMATTED=$(echo "scale=1; $NET_WORTH / 1000000" | bc -l)
                echo "   Estimated net worth: \$${FORMATTED}M"
            else
                echo "   Estimated net worth: \$${NET_WORTH}"
            fi
        fi
        
        echo "   Top matches:"
        echo "$RESPONSE" | grep -o '"name":"[^"]*"' | head -3 | sed 's/"name":"//g' | sed 's/"$//g' | sed 's/^/     - /'
    else
        echo "Prediction failed!"
        echo "Response: $RESPONSE"
        exit 1
    fi
else
    echo "No test image found, skipping prediction test"
    echo "Available test images should be in test_images/ folder"
fi

echo ""
echo "API testing complete!"
echo "Your API is working at: $API_URL" 