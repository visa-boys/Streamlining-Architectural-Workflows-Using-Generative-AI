# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load the layout image
# img_path = "kp_layout.png"
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# # Apply Canny edge detection
# edges = cv2.Canny(img, 100, 200)

# # Create a binary array based on edge detection
# binary_array = np.where(edges != 0, 1, 0)

# # Visualize the binary array
# plt.figure(figsize=(8, 8))
# plt.imshow(binary_array, cmap='binary', interpolation='nearest')
# plt.axis('off')
# plt.show()

# np.save("kp_layout_boundary_array.npy", binary_array)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the layout image
# img_path = "kp_layout.png"
# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# # Define a threshold for non-white color
# color_threshold = 200  # Adjust as needed

# # Create a binary array based on non-white color detection
# binary_array_color = np.where(img > color_threshold, 1, 0)

# # Visualize the binary array for non-white color detection
# plt.figure(figsize=(8, 8))
# plt.imshow(binary_array_color, cmap='binary', interpolation='nearest')
# plt.axis('off')
# plt.show()

# # Save the binary array for non-white color detection
# np.save("kp_layout_color_array.npy", binary_array_color)

def save_binary_array(img_path):
    # Read the image in grayscale
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Define a threshold for non-white color
    color_threshold = 200  # Adjust as needed

    # Create a binary array based on non-white color detection
    binary_array_color = np.where(img > color_threshold, 1, 0)

    # Visualize the binary array for non-white color detection
    plt.figure(figsize=(8, 8))
    plt.imshow(binary_array_color, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.show()

    # Save the binary array for non-white color detection
    np.save("kp_layout_color_array.npy", binary_array_color)



import numpy as np
import matplotlib.pyplot as plt

def visualize_binary_array(npy_file_path):
    # Load the binary array from the npy file
    binary_array = np.load(npy_file_path)

    # Visualize the binary array
    plt.figure(figsize=(8, 8))
    plt.imshow(binary_array, cmap='binary', interpolation='nearest')
    plt.axis('off')
    plt.show()

# Example usage:
npy_file_path = "/Users/arunachalamm/Desktop/layout.npy"
visualize_binary_array(npy_file_path)
