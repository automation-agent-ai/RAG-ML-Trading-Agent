#!/bin/bash

# Enhanced RAG Pipeline Deployment Script
# Deploys to trading-agent.automation-agent.ai

set -e  # Exit on any error

echo "🚀 Starting Enhanced RAG Pipeline Deployment..."
echo "=================================================="

# Configuration
DOMAIN="trading-agent.automation-agent.ai"
CONTAINER_NAME="trading-agent-container"
IMAGE_NAME="trading-agent:latest"
NGINX_CONFIG="/etc/nginx/sites-available/trading-agent.conf"
NGINX_ENABLED="/etc/nginx/sites-enabled/trading-agent.conf"
PORT=8300

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)"
    exit 1
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    log_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    log_error "Docker Compose is not available. Please install Docker Compose."
    exit 1
fi

# Stop existing container if running
log_info "Stopping existing container if running..."
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    docker stop $CONTAINER_NAME
    log_success "Stopped existing container"
else
    log_info "No existing container found"
fi

# Remove existing container
log_info "Removing existing container if exists..."
if docker ps -aq -f name=$CONTAINER_NAME | grep -q .; then
    docker rm $CONTAINER_NAME
    log_success "Removed existing container"
fi

# Build the Docker image
log_info "Building Docker image..."
docker build -t $IMAGE_NAME . --no-cache
log_success "Docker image built successfully"

# Create data directories if they don't exist
log_info "Ensuring data directories exist..."
mkdir -p data models
chmod 755 data models
log_success "Data directories ready"

# Start the container using docker-compose
log_info "Starting Enhanced RAG Pipeline container..."
if command -v docker-compose &> /dev/null; then
    docker-compose -f docker-compose.yml up -d
else
    docker compose -f docker-compose.yml up -d
fi

# Wait for container to be healthy
log_info "Waiting for container to be healthy..."
max_attempts=30
attempt=1
while [[ $attempt -le $max_attempts ]]; do
    if docker exec $CONTAINER_NAME curl -f http://localhost:$PORT/api/health &> /dev/null; then
        log_success "Container is healthy and responding"
        break
    fi
    if [[ $attempt -eq $max_attempts ]]; then
        log_error "Container failed to become healthy within 5 minutes"
        docker logs $CONTAINER_NAME --tail 50
        exit 1
    fi
    echo -n "."
    sleep 10
    ((attempt++))
done

# Configure Nginx
log_info "Configuring Nginx..."

# Copy nginx configuration
cp trading-agent.conf $NGINX_CONFIG
log_success "Nginx configuration copied"

# Enable the site
if [[ -f $NGINX_ENABLED ]]; then
    rm $NGINX_ENABLED
fi
ln -s $NGINX_CONFIG $NGINX_ENABLED
log_success "Nginx site enabled"

# Test nginx configuration
if nginx -t; then
    log_success "Nginx configuration is valid"
else
    log_error "Nginx configuration is invalid"
    exit 1
fi

# Check if SSL certificates exist, if not, provide instructions
if [[ ! -f "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" ]]; then
    log_warning "SSL certificates not found for $DOMAIN"
    log_info "First, ensure the container is running and accessible on port $PORT"
    log_info "Then set up SSL certificates with:"
    echo "   sudo certbot --nginx -d $DOMAIN"
    log_info "Note: Domain must be accessible via HTTP first for SSL verification"
    
    # Create temporary HTTP-only config
    log_info "Creating temporary HTTP-only configuration..."
    cat > $NGINX_CONFIG << EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://localhost:$PORT;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
        client_max_body_size 100M;
    }
}
EOF
    log_warning "Temporary HTTP-only configuration created"
fi

# Reload Nginx
log_info "Reloading Nginx..."
systemctl reload nginx
log_success "Nginx reloaded successfully"

# Final status check
log_info "Performing final status check..."
if curl -f http://localhost:$PORT/api/health &> /dev/null; then
    log_success "Application is responding on port $PORT"
else
    log_error "Application is not responding on port $PORT"
    docker logs $CONTAINER_NAME --tail 20
    exit 1
fi

echo ""
echo "=================================================="
log_success "🎉 Enhanced RAG Pipeline Deployed Successfully!"
echo "=================================================="
echo ""
echo "📊 Deployment Details:"
echo "   🌐 Domain: $DOMAIN"
echo "   🔗 URL: http://$DOMAIN (HTTPS after SSL setup)"
echo "   🐳 Container: $CONTAINER_NAME"
echo "   🚪 Port: $PORT"
echo "   📋 API Docs: http://$DOMAIN/docs"
echo "   ❤️  Health: http://$DOMAIN/api/health"
echo ""
echo "🔧 Next Steps:"
echo "   1. Set up SSL: sudo certbot --nginx -d $DOMAIN"
echo "   2. Configure OpenAI API key in .env file"
echo "   3. Upload training data if needed"
echo ""
echo "📝 Useful Commands:"
echo "   • View logs: docker logs $CONTAINER_NAME -f"
echo "   • Restart: docker restart $CONTAINER_NAME"
echo "   • Stop: docker stop $CONTAINER_NAME"
echo "   • Shell access: docker exec -it $CONTAINER_NAME bash"
echo ""
log_success "Deployment complete! 🚀"