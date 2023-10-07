from PIL import Image

def resize_image(image_path, output_path, max_size):
    # Open the image file
    image = Image.open(image_path)

    # Calculate the new size while maintaining the aspect ratio
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * max_size / width)
    else:
        new_height = max_size
        new_width = int(width * max_size / height)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    # Save the resized image
    resized_image.save(output_path)

# Example usage
input_image_path = 'windows-mouse-cursor-png-2.png'
output_image_path = 'cursor.png'
max_size = 20  # Maximum size (width or height) for the resized image

resize_image(input_image_path, output_image_path, max_size)
