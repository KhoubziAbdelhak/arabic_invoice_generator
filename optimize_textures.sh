#!/bin/bash

# Set paths
TEXTURE_DIR="data/textures"
OUTPUT_DIR="data/textures_optimized"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Target size for the long edge (in pixels)
TARGET_SIZE=1500

# JPEG quality (0-100, lower means smaller file size)
QUALITY=75

# Process each texture file
for texture in "$TEXTURE_DIR"/*.jpg; do
    filename=$(basename "$texture")
    output_file="$OUTPUT_DIR/$filename"
    
    echo "Processing $filename..."
    
    # Get dimensions
    width=$(identify -format "%w" "$texture")
    height=$(identify -format "%h" "$texture")
    
    # Calculate scaling factor based on the longer dimension
    if [ $width -gt $height ]; then
        scale="$TARGET_SIZE:-1"
    else
        scale="-1:$TARGET_SIZE"
    fi
    
    # Use ffmpeg to resize and compress
    ffmpeg -i "$texture" -vf "scale=$scale" -q:v $((100-QUALITY)) "$output_file" -y
    
    # Print size reduction
    original_size=$(du -k "$texture" | cut -f1)
    new_size=$(du -k "$output_file" | cut -f1)
    reduction=$((100 - (new_size * 100 / original_size)))
    
    echo "Reduced $filename by $reduction% (${original_size}KB → ${new_size}KB)"
done

echo "Texture optimization complete. Optimized textures are in $OUTPUT_DIR"