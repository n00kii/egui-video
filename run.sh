#!/bin/bash

# egui-video run script for macOS
# This script sets up the environment and runs the example application

set -e  # Exit on any error

echo "🚀 Starting egui-video example..."

# Set up environment variables for macOS
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export LIBRARY_PATH="/opt/homebrew/lib:$LIBRARY_PATH"

echo "📦 Environment variables set:"
echo "   PKG_CONFIG_PATH: $PKG_CONFIG_PATH"
echo "   LIBRARY_PATH: $LIBRARY_PATH"

# Check if we're in the right directory
if [ ! -f "Cargo.toml" ]; then
    echo "❌ Error: Cargo.toml not found. Please run this script from the egui-video project root."
    exit 1
fi

# Check if dependencies are installed
if ! command -v pkg-config &> /dev/null; then
    echo "❌ Error: pkg-config not found. Please install it with: brew install pkg-config"
    exit 1
fi

if ! command -v ffmpeg &> /dev/null; then
    echo "❌ Error: ffmpeg not found. Please install it with: brew install ffmpeg"
    exit 1
fi

echo "✅ Dependencies check passed"

# Run the example
echo "🔨 Building and running example..."
echo "   Using --release for optimal performance"
echo "   Using --no-default-features to avoid SDL2 compilation issues"

cargo run --example main --release --no-default-features

echo "👋 Example finished" 