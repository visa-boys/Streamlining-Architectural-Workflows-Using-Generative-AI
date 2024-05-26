import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize(rectangles,labels):
    # Create figure and axis
    fig, ax = plt.subplots()

    # Set limits
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)

    # Add rectangles to the plot
    for rect, label in zip(rectangles, labels):
        ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2]-rect[0], rect[3]-rect[1], linewidth=1, edgecolor='r', facecolor='none'))
        # Calculate center coordinates of the rectangle
        center_x = (rect[0] + rect[2]) / 2
        center_y = (rect[1] + rect[3]) / 2
        # Add label at the center
        ax.text(center_x, center_y, label, fontsize=8, color='black', ha='center', va='center')  # Center the text

    ax.set_aspect('equal')
    # Display the plot
    plt.axis('off')
    plt.title('Room Type Suggestion')
    plt.show()
