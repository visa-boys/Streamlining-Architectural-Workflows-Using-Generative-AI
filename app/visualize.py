import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as colors

def visualize(rectangles, labels):
    # Create figure and axis with a smaller size
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the size as needed

    # Set limits
    ax.set_xlim(0, 256)
    ax.set_ylim(0, 256)

    # Define a list of mild colors from the BASE_COLORS dictionary
    mild_colors = ['lightskyblue', 'lightgreen', 'lightsalmon', 'khaki', 'lavender', 'lightcyan', 'lightcoral', 'wheat', 'plum', 'mistyrose', 'lightgoldenrodyellow']

    # Add rectangles to the plot with different colors
    legend_patches = []
    color_map = {}
    for rect, label in zip(rectangles, labels):
        if label not in color_map:
            color_map[label] = mild_colors[len(color_map) % len(mild_colors)]
        color = color_map[label]
        rect_patch = ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0], rect[3] - rect[1],
                                                    linewidth=1, edgecolor='black', facecolor=color))
        legend_patches.append(rect_patch)

    ax.set_aspect('equal')

    # Display the plot
    plt.axis('off')
    #plt.title('Room Type Suggestions from Visual Prompts')

    # Add legend with unique labels
    unique_labels = list(set(labels))
    plt.legend([legend_patches[labels.index(label)] for label in unique_labels], unique_labels, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

    return fig  # Return the figure explicitly