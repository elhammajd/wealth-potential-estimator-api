#!/bin/bash

# Simple deployment script for the wealth estimator API
# Handles local Docker testing and prep for cloud deployment

set -e

echo "Wealth Estimator API - Deploy Script"
echo "======================================="

# Build the Docker image
build_image() {
    echo "Building Docker image..."
    docker build -t wealth-estimator-api .
    echo "Built successfully!"
}

# Run it locally
run_local() {
    echo "Starting API on port 8000..."
    
    # Kill any existing container
    if docker ps -q -f name=wealth-api > /dev/null; then
        echo "Stopping old container..."
        docker stop wealth-api
        docker rm wealth-api
    fi
    
    # Start fresh container
    docker run -d \
        -p 8000:8000 \
        -e WEALTHAPI_PRETRAINED=1 \
        --name wealth-api \
        wealth-estimator-api
    
    # Wait a moment for startup
    sleep 3
    
    # Check if it's working
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "API is running!"
        echo "Try it: http://localhost:8000"
        echo ""
        echo "Test with:"
        echo "curl -X POST \"http://localhost:8000/predict\" -H \"Content-Type: multipart/form-data\" -F \"file=@test_images/business_person.jpg\""
    else
        echo "Something went wrong. Check logs with: docker logs wealth-api"
    fi
}

# Stop the local container
stop_local() {
    if docker ps -q -f name=wealth-api > /dev/null; then
        echo "Stopping API..."
        docker stop wealth-api
        docker rm wealth-api
        echo "Stopped!"
    else
        echo "Nothing running."
    fi
}

# Show deployment options
show_deploy_options() {
    echo "Free Hosting Deployment Options:"
    echo ""
    echo "1. Railway (Recommended for demos):"
    echo "   - Go to railway.app"
    echo "   - Connect your GitHub account"
    echo "   - Deploy from GitHub repo"
    echo "   - Set WEALTHAPI_PRETRAINED=1 in environment variables"
    echo ""
    echo "2. Render:"
    echo "   - Go to render.com"
    echo "   - Create new Web Service from GitHub"
    echo "   - Use the render.yaml config"
    echo ""
    echo "3. Fly.io:"
    echo "   - Install flyctl"
    echo "   - Run: fly deploy"
    echo "   - Uses fly.toml config"
    echo ""
    echo "4. Google Cloud Run:"
    echo "   - Build: docker build -t gcr.io/PROJECT/wealth-api ."
    echo "   - Push: docker push gcr.io/PROJECT/wealth-api"
    echo "   - Deploy: gcloud run deploy --image gcr.io/PROJECT/wealth-api"
}

# Test a deployed API
test_deployment() {
    URL=$1
    if [ -z "$URL" ]; then
        echo "Usage: $0 test https://your-api-url.com"
        exit 1
    fi
    
    echo "Testing deployment at: $URL"
    
    # Health check
    if curl -s "$URL/health" | grep -q "healthy"; then
        echo "Health check passed!"
    else
        echo "Health check failed!"
        exit 1
    fi
    
    # Try prediction if we have a test image
    if [ -f "test_images/business_person.jpg" ]; then
        echo "Testing prediction endpoint..."
        curl -X POST "$URL/predict" \
            -H "Content-Type: multipart/form-data" \
            -F "file=@test_images/business_person.jpg"
        echo "Predict endpoint working!"
    else
        echo "No test image found, skipping predict test"
    fi
}

# Main command handling
case "$1" in
    "build")
        build_image
        ;;
    "run")
        build_image
        run_local
        ;;
    "stop")
        stop_local
        ;;
    "deploy")
        show_deploy_options
        ;;
    "test")
        test_deployment $2
        ;;
    *)
        echo "Usage: $0 {build|run|stop|deploy|test}"
        echo ""
        echo "Commands:"
        echo "  build  - Build Docker image"
        echo "  run    - Build and run locally on port 8000"
        echo "  stop   - Stop local container"
        echo "  deploy - Show deployment options"
        echo "  test   - Test deployed API (requires URL)"
        ;;
esac 