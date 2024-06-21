from quiz import image_caption_from_path

image_path = "test_spec/image/test_03.png"
print("Image path: " + image_path)
description = image_caption_from_path(image_path)

print("Description:", description)
