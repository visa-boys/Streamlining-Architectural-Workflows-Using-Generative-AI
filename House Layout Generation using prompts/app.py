from flask import Flask, render_template, request
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import google.generativeai as genai
from dotenv import load_dotenv
import os

app = Flask(__name__)

# Load environment variables
load_dotenv()
load_model = os.environ.get("load_model")
genai.configure(api_key=load_model)
ner_model = genai.GenerativeModel(os.environ.get("ner_model"))

# Function to generate bounding box with direction
def generate_bounding_box_with_direction(min_x, max_x, min_y, max_y, direction, width, height):
    center_x = (max_x + min_x) / 2
    center_y = (max_y + min_y) / 2
    
    if direction == "North":
        y1 = center_y
        y2 = min(center_y + height, max_y)
        x1 = max(center_x - width / 2, min_x)
        x2 = min(center_x + width / 2, max_x)
    elif direction == "North east":
        y1 = center_y
        y2 = min(center_y + height, max_y)
        x1 = center_x
        x2 = min(center_x + width, max_x)
    elif direction == "East":
        y1 = max(center_y - height / 2, min_y)
        y2 = min(center_y + height / 2, max_y)
        x1 = center_x
        x2 = min(center_x + width, max_x)
    elif direction == "South east":
        y1 = max(center_y - height, min_y)
        y2 = center_y
        x1 = center_x
        x2 = min(center_x + width, max_x)
    elif direction == "South":
        y1 = max(center_y - height, min_y)
        y2 = center_y
        x1 = max(center_x - width / 2, min_x)
        x2 = min(center_x + width / 2, max_x)
    elif direction == "South west":
        y1 = max(center_y - height, min_y)
        y2 = center_y
        x1 = max(center_x - width, min_x)
        x2 = center_x
    elif direction == "West":
        y1 = max(center_y - height / 2, min_y)
        y2 = min(center_y + height / 2, max_y)
        x1 = max(center_x - width, min_x)
        x2 = center_x
    elif direction == "North west":
        y1 = center_y
        y2 = min(center_y + height, max_y)
        x1 = max(center_x - width, min_x)
        x2 = center_x
    
    return (x1, y1, x2, y2)

# Function to generate non-overlapping bounding boxes
def generate_non_overlapping_bounding_boxes(room_data, min_x, max_x, min_y, max_y, min_distance):
    bounding_boxes = []
    centers = []
    
    for room in room_data:
        room_name = room['name']
        width = room['width']
        height = room['height']
        direction = room['direction']
        color = room['color']
        
        while True:
            bounding_box = generate_bounding_box_with_direction(min_x, max_x, min_y, max_y, direction, width, height)
            center_x = (bounding_box[0] + bounding_box[2]) / 2
            center_y = (bounding_box[1] + bounding_box[3]) / 2
            new_center = (center_x, center_y)
            
            if not check_center_collision(new_center, centers, min_distance):
                bounding_boxes.append({'bbox': bounding_box, 'color': color, 'name': room_name})
                centers.append(new_center)
                break
    
    return bounding_boxes

# Function to check center collision
def check_center_collision(new_center, existing_centers, min_distance):
    for center in existing_centers:
        distance = ((new_center[0] - center[0]) ** 2 + (new_center[1] - center[1]) ** 2) ** 0.5
        if distance < min_distance:
            return True
    return False

# Function to visualize bounding boxes and save as image
def visualize_bounding_boxes(bounding_boxes, min_x, max_x, min_y, max_y):
    # Set non-interactive backend
    matplotlib.use('Agg')
    
    fig, ax = plt.subplots()
    room_colors = {}  # To store room colors for legend
    
    for room in bounding_boxes:
        bounding_box = room['bbox']
        color = room['color']
        room_name = room['name']
        
        # Plot bounding box with specified color
        x_min, y_min, x_max, y_max = bounding_box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor=color, label=room_name)
        ax.add_patch(rect)
        
        # Add room name as annotation
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Store room colors for legend
        room_colors[room_name] = color
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Non-Overlapping Bounding Boxes')
    
    # Add legend
    handles = [patches.Patch(color=color, label=room_name) for room_name, color in room_colors.items()]
    plt.legend(handles=handles, loc='upper right', title='Rooms', fontsize='small')
    
    plt.grid(True)
    
    # Save the plot as an image file
    img_path = 'static/layout.png'  # Path to save the image in the 'static' folder
    plt.savefig(img_path)
    
    # Close the plot to release resources
    plt.close()
    
    return img_path

# Function to convert generated content to room data
def convert_to_room_data(generated_content):
    room_data = []

    # Split the generated content by newlines
    lines = generated_content.split('\n')

    for line in lines:
        # Check if the line starts with "Room:" to extract room name
        if line.startswith('Room:'):
            room_name = line.split('Room: ')[-1].strip()
        
        # Check if the line starts with "Dimension:" to extract width and height
        elif line.startswith('Dimension:'):
            dimensions = line.split('Dimension: ')[-1].strip().split(' x ')
            width = float(dimensions[0].split()[0])
            height = float(dimensions[1].split()[0])
        
        # Check if the line starts with "Direction:" to extract direction
        elif line.startswith('Direction:'):
            direction = line.split('Direction: ')[-1].strip()

            # Map direction to compass directions for consistency
            direction_mapping = {
                "north": "North",
                "south": "South",
                "east": "East",
                "west": "West",
                "northeast": "North east",
                "northwest": "North west",
                "southeast": "South east",
                "southwest": "South west"
            }
            direction = direction_mapping.get(direction.lower(), "Unknown")

            # Assign a random color for now
            color = "#{:06x}".format(random.randint(0, 0xFFFFFF))

            # Add the room data to the list
            room_data.append({'name': room_name, 'width': width, 'height': height, 'direction': direction, 'color': color})
    
    return room_data

# Define boundaries
min_x = 0
max_x = 100
min_y = 0
max_y = 100
min_distance = 10  # Minimum distance between centers
room_data = []  # List to store room data

@app.route('/', methods=['GET', 'POST'])
def index():
    global room_data
    
    if request.method == 'POST':
        room_name = request.form['room_name']
        width = float(request.form['width'])
        height = float(request.form['height'])
        direction = request.form['direction']
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))  # Random color
        
        room_data.append({'name': room_name, 'width': width, 'height': height, 'direction': direction, 'color': color})
    
    return render_template('index.html', room_data=room_data)

@app.route('/generate_layout', methods=['POST'])
def generate_layout():
    global room_data
    bounding_boxes = generate_non_overlapping_bounding_boxes(room_data, min_x, max_x, min_y, max_y, min_distance)
    img_path = visualize_bounding_boxes(bounding_boxes, min_x, max_x, min_y, max_y)
    print(room_data)
    # Clear room_data after generating layout
    room_data = []
    
    return render_template('layout.html', img_path=img_path)

@app.route('/prompt', methods=['GET'])
def prompt():
    return render_template('prompt.html')

@app.route('/getroomdata', methods=['POST'])
def gemini_generate():
    global room_data
    
    if request.method == 'POST':
        prompt_text = request.form['prompt']
        
        # Load prefix from file
        prefix_file = "preprocess.txt"
        with open(prefix_file, "r") as p:
            prefix = p.read()

        # Append prompt text to prefix
        s = prompt_text

        # Generate content using Gemini API
        response = ner_model.generate_content(prefix + s)

        # Extract room data from Gemini generated content
        generated_content = response.text
        print(generated_content)

        # Convert generated content to room data
        room_data = convert_to_room_data(generated_content)
        print(room_data)

        bounding_boxes = generate_non_overlapping_bounding_boxes(room_data, min_x, max_x, min_y, max_y, min_distance)

        # Visualize and save the layout as an image
        img_path = visualize_bounding_boxes(bounding_boxes, min_x, max_x, min_y, max_y)

        # Clear room_data after generating layout
        room_data = []

        return render_template('layout.html', img_path=img_path)

if __name__ == '__main__':
    app.run(debug=True,port=5001)
