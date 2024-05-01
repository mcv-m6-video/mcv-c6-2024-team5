import imageio
import os

content_for_GIFs_path = 'content_for_GIFs/'

# In this folder we have 4 types of images:
# 1. The original frames - frame{i:05d}.jpg
# 2. The RGB difference images - rgb_diff_{i:05d}.jpg
# 3. The skeletons images - skeletons_{i:05d}.jpg
# 4. The optical flow images - flow_{i:05d}.png

# We will generate GIFs for each type of image
for image_type in ['frame', 'rgb_diff_', 'skeletons_', 'flow_']:
    if image_type == 'flow_':
        file_type = '.png'
    else:
        file_type = '.jpg'
    # Get the images for the current type
    images = [f for f in os.listdir(content_for_GIFs_path) if f.endswith(file_type) and f.startswith(f'{image_type}')]
    
    print(f'Generating GIF for {image_type}...')

    # Sort the images
    images.sort()
    
    # Create the GIF
    with imageio.get_writer(content_for_GIFs_path + f'{image_type}.gif', mode='I') as writer:
        for image in images:
            writer.append_data(imageio.imread(content_for_GIFs_path + image))