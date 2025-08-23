from PIL import Image
from collections import Counter

def create_global_palette(images, max_colors=256):
    """Generate a global color palette from all images."""
    all_pixels = []
    for img in images:
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Sample pixels to reduce memory usage
        all_pixels.extend(img.getdata()[::20])  # Adjust sampling as needed
    # Get most common colors
    color_counts = Counter(all_pixels)
    return [color for color, _ in color_counts.most_common(max_colors)]

def quantize_image(img, palette):
    """Quantize an image to the given palette."""
    pal_img = Image.new('P', (1, 1))
    pal_img.putpalette([c for color in palette for c in color])
    return img.convert('RGB').quantize(palette=pal_img, dither=Image.Dither.NONE)

def create_gif(image_paths, timecodes, output_path):
    # Load images
    images = [Image.open(path) for path in image_paths]
    
    # Ensure consistent sizes
    base_size = images[0].size
    images = [img if img.size == base_size else img.resize(base_size) 
              for img in images]
    
    # Generate and apply global palette
    global_palette = create_global_palette(images)
    images_quantized = [quantize_image(img, global_palette) for img in images]
    
    # Convert timecodes to durations (milliseconds)
    durations = [
        int((timecodes[i+1] - timecodes[i]) * 1000) 
        for i in range(len(timecodes)-1)
    ]
    durations.append(100)  # Last frame duration (default: 100ms)
    
    # Save optimized GIF
    images_quantized[0].save(
        output_path,
        save_all=True,
        append_images=images_quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,  # Restore to background for delta compression
        palette=global_palette
    )

def test():
    # Example Usage
    image_paths = ["frame1.png", "frame2.png", "frame3.png"]  # Your image paths
    timecodes = [0.0, 0.5, 1.2]  # Timestamps in seconds
    create_gif(image_paths, timecodes, "output.gif")

if __name__ == "__main__":
    test()