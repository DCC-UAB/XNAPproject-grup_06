# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:38:34 2024

@author: alexg
"""

import os
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter

def get_image_resolutions(directory):
    resolutions = []
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            try:
                with Image.open(os.path.join(directory, filename)) as img:
                    resolutions.append(img.size)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        print(count)
        count+=1
        if count ==10000:
            break
    return resolutions

def plot_common_resolutions(directory):
    resolutions = get_image_resolutions(directory)
    resolution_counts = Counter(resolutions)
    common_resolutions = resolution_counts.most_common()

    if not common_resolutions:
        print("No images found in the directory or unable to process images.")
        return

    resolutions, counts = zip(*common_resolutions)

    # Convert resolutions to string for better display in the plot
    resolution_strings = [f"{w}x{h}" for w, h in resolutions]
    
    plt.figure(figsize=(10, 6))
    plt.bar(resolution_strings[0:10], counts[0:10])
    plt.xlabel('Resolution')
    plt.ylabel('Count')
    plt.title('Most Common Image Resolutions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



directory = "imatges_proc/"
plot_common_resolutions(directory)
