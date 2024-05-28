import streamlit as st
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import google.generativeai as genai
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import json
from io import BytesIO

# Load environment variables
load_dotenv()
load_model = os.environ.get("load_model")
genai.configure(api_key=load_model)
ner_model = genai.GenerativeModel(os.environ.get("ner_model"))

#converting to npy

# def save_binary_array(img_path):
#     # Read the image in grayscale
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

#     # Define a threshold for non-white color
#     color_threshold = 200  # Adjust as needed

#     # Create a binary array based on non-white color detection
#     binary_array_color = np.where(img > color_threshold, 1, 0)

#     # Visualize the binary array for non-white color detection
#     # plt.figure(figsize=(8, 8))
#     # plt.imshow(binary_array_color, cmap='binary', interpolation='nearest')
#     # plt.axis('off')
#     # plt.show()

#     # Save the binary array for non-white color detection
#     np.save("kp_final_array.npy", binary_array_color)

# Function to save binary array
def save_binary_array(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    color_threshold = 200  # Adjust as needed
    binary_array_color = np.where(img > color_threshold, 1, 0)
    np.save("kp_final_array.npy", binary_array_color)

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
    max_attempts=100
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
                bounding_boxes.append({'bbox': bounding_box, 'color': color, 'name': room_name,'direction':direction})
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

def visualize_bounding_boxes(bounding_boxes, min_x, max_x, min_y, max_y):
    # Set non-interactive backend
    plt.switch_backend('agg')
    
    fig, ax = plt.subplots()
    room_colors = {}  # To store room colors for legend
    
    # Plot boundaries first
    for room in bounding_boxes:
        bounding_box = room['bbox']
        color = room['color']
        room_name = room['name']
        
        # Plot bounding box outline with specified color
        x_min, y_min, x_max, y_max = bounding_box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor='none', label=room_name)
        ax.add_patch(rect)
        
        # Store room colors for legend
        room_colors[room_name] = color

    # Save the plot with only boundaries
    postprocess_path = 'kp_layout.png'
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_title('Boundary Detection')

    # Remove axis labels and ticks for boundary plot
    ax.set_xticks([])
    ax.set_yticks([])
    ax.xaxis.set_tick_params(size=0)
    ax.yaxis.set_tick_params(size=0)
    ax.set_frame_on(False)
    
    # Save the plot as an image file for boundary detection and convert to npy
    plt.savefig(postprocess_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    save_binary_array(postprocess_path)

    # Plot filled boxes on top of boundaries
    fig, ax = plt.subplots()
    
    for room in bounding_boxes:
        bounding_box = room['bbox']
        color = room['color']
        room_name = room['name']
        
        # Plot filled bounding box with specified color
        x_min, y_min, x_max, y_max = bounding_box
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor=color, label=room_name)
        ax.add_patch(rect)
    
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Generated House Layout')

    # Add legend
    handles = [patches.Patch(color=color, label=room_name) for room_name, color in room_colors.items()]
    ax.legend(handles=handles, loc='upper right', title='Rooms', fontsize='small')

    # Save the plot with filled boxes
    img_path = 'layout.png'
    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
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

# Function to validate layout
def validate_layout(room_data):
    errors = []
    for room in room_data:
        room_name = room['name']
        width = room['width']
        height = room['height']
        area = width * height
        
        if room_name.lower() == 'kitchen':
            if area < 5 or width < 1.8:
                errors.append("The kitchen fails to conform to the specifications outlined in the Tamil Nadu Combined Development and Building Rules (TNCDRBR). Specifically, it does not meet the requirement stipulating that the area of a kitchen with a separate dining area must not be less than 5.0 square meters, with a minimum width of 1.8 meters.")
        elif room_name.lower() == 'bathroom':
            if area < 1.4 or width < 1:
                errors.append("The bathroom fails to conform to the specifications outlined in the Tamil Nadu Combined Development and Building Rules (TNCDRBR). Specifically, it does not meet the requirement stipulating that the area of a bathroom must not be less than 1.4 square meters, with a minimum width of 1 meters.")
        else:
            if area < 7.5 or width < 2.4:
                errors.append("The layout fails to conform to the specifications outlined in the Tamil Nadu Combined Development and Building Rules (TNCDRBR). Specifically, it does not meet the requirement stipulating that the area of a room must not be less than 7.5 square meters, with a minimum width of 2.4 meters.")
    
    return errors

# Define boundaries
min_x = 0
max_x = 100
min_y = 0
max_y = 100
min_distance = 5  # Minimum distance between centers

# Streamlit app
def main():
    st.title("House Layout Generator")
    st.sidebar.title("Menu Driven House Layout Generation")

    # Initialize session state
    if 'room_data' not in st.session_state:
        st.session_state.room_data = []

    # Sidebar for adding rooms
    st.sidebar.header("Add Room")
    room_name = st.sidebar.text_input("Room Name")
    width = st.sidebar.number_input("Width", min_value=1.0, step=1.0)
    height = st.sidebar.number_input("Length", min_value=1.0, step=1.0)
    direction = st.sidebar.selectbox("Direction", ["North", "North east", "East", "South east", "South", "South west", "West", "North west"])
    
    if st.sidebar.button("Add"):
        color = "#{:06x}".format(random.randint(0, 0xFFFFFF))  # Random color
        st.session_state.room_data.append({'name': room_name, 'width': width, 'height': height, 'direction': direction, 'color': color})
        st.success("Room Added!")
    
    if st.sidebar.button("Clear All Rooms"):
        st.session_state.room_data.clear()
        st.warning("All Rooms Cleared!")
    
    # Main content
    if st.button("Generate Layout"):
        if not st.session_state.room_data:
            st.warning("No rooms added. Please add rooms first.")
        else:
            bounding_boxes = generate_non_overlapping_bounding_boxes(st.session_state.room_data, min_x, max_x, min_y, max_y, min_distance)
            img_path = visualize_bounding_boxes(bounding_boxes, min_x, max_x, min_y, max_y)
            st.image(img_path, caption='Generated House Layout', use_column_width=True)
            st.success("Layout Generated!")
            # Download the npy file when the "Generate Layout" button is clicked
            binary_data = np.load("kp_final_array.npy")
            with BytesIO() as buffer:
                np.save(buffer, binary_data)
                buffer.seek(0)
                st.download_button(label="Download Layout.npy", data=buffer, file_name="layout.npy", mime="application/octet-stream")

            ### json
            with st.spinner("Creating JSON..."):
                json_data = json.dumps(bounding_boxes)
            with BytesIO() as json_buffer:
                json_buffer.write(json_data.encode())
                json_buffer.seek(0)
                st.download_button(label="Download Layout.json", data=json_buffer, file_name="layout.json", mime="application/json")



    # Validate Layout Button
    if st.button("Validate Layout"):
        if not st.session_state.room_data:
            st.warning("No rooms added. Please add rooms first.")
        else:
            errors = validate_layout(st.session_state.room_data)
            if errors:
                for error in errors:
                    st.error(error)
            else:
                st.success("Layout is valid!")


    prefix_file = "preprocess.txt"
    with open(prefix_file, "r") as p:
        prefix = p.read()

    st.sidebar.header("House Layout Generation using Prompts")
    prompt_text = st.sidebar.text_area("Enter Prompt", height=100)
    
    if st.sidebar.button("Generate House Layout"):
        if not prompt_text:
            st.warning("Please enter a prompt.")
        else:
            response = ner_model.generate_content(prefix+prompt_text)
            generated_content = response.text
            room_data = convert_to_room_data(generated_content)
            print(generated_content)
            if room_data:
                bounding_boxes = generate_non_overlapping_bounding_boxes(room_data, min_x, max_x, min_y, max_y, min_distance)
                img_path = visualize_bounding_boxes(bounding_boxes, min_x, max_x, min_y, max_y)
                st.image(img_path, caption='Generated House Layout', use_column_width=True)
                st.success("House Layout has been Generated")
                binary_data = np.load("kp_final_array.npy")
                with BytesIO() as buffer:
                    np.save(buffer, binary_data)
                    buffer.seek(0)
                    st.download_button(label="Download Layout.npy", data=buffer, file_name="layout.npy", mime="application/octet-stream")
                
                ### json
                with st.spinner("Creating JSON..."):
                    json_data = json.dumps(bounding_boxes)
                with BytesIO() as json_buffer:
                    json_buffer.write(json_data.encode())
                    json_buffer.seek(0)
                    st.download_button(label="Download Layout.json", data=json_buffer, file_name="layout.json", mime="application/json")
                
            else:
                st.error("Error occurred!!")
    
if __name__ == "__main__":
    main()
