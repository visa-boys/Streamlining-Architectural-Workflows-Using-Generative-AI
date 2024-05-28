import streamlit as st
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import random
import matplotlib.pyplot as plt
import re
import matplotlib.patches as patches
import matplotlib
from streamlit_lottie import st_lottie
import os
import json

# preload
load_dotenv()
load_model = os.environ.get("load_model")
genai.configure(api_key=load_model)
model = genai.GenerativeModel(os.environ.get("ner_model"))
prefix_file = "preprocess.txt"
with open(prefix_file, "r") as p:
    prefix = p.read()
model_avatar = "avatar.png"

temp_reply=""

validate_text = ""
with open("validate.txt", "r") as validate_file:
    validate_text = validate_file.read()

# Global variables
if "room_data" not in st.session_state:
    st.session_state.room_data = []

min_x = 0
max_x = 100
min_y = 0
max_y = 100
min_distance = 10

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
        height = room['length']
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

# Function to convert generated content to room data
def convert_to_room_data(generated_content):
    rooms = []
    
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
            rooms.append({'name': room_name, 'width': width, 'length': height, 'direction': direction, 'color': color})
        else:
            temp_reply=line.split("\n")[-1]
    return rooms

def execute_prompt(prompt):
    response = model.generate_content(prompt)
    return response.text


# Function to generate layout using current room data
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
    
    # Save the plot as an image file for boundary detection
    plt.savefig(postprocess_path, bbox_inches='tight', pad_inches=0)
    plt.close()

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

def generate_layout(room_data):
    
    if not room_data:
        st.warning("No rooms added. Please add rooms first.")
    else:
        bounding_boxes = generate_non_overlapping_bounding_boxes(room_data, min_x, max_x, min_y, max_y, min_distance)
        img_path = visualize_bounding_boxes(bounding_boxes, min_x, max_x, min_y, max_y)
        st.image(img_path, caption='Generated House Layout', use_column_width=True)
        st.success("Layout Generated!")

# def main():
#     st.title("ARCHI BOT")
    
#     # Initialize session state for room data and messages
#     if 'room_data' not in st.session_state:
#         st.session_state.room_data = []
    
#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     # React to user input
#     if prompt := st.chat_input("What's up"):
        
#         # Display user message in chat message container
#         st.session_state.messages.append({"role": "user", "parts": [prompt]})

#          # Determine if "validate" is present in the prompt
#         if re.search(r'\bvalidate\b', prompt, re.IGNORECASE):
#             prompt = validate_text + prompt + str(st.session_state.room_data)
#         else:
#             prompt = prefix + prompt + str(st.session_state.room_data)
        
#         # Generate response from Gemini API
#         if st.session_state.messages:
            
#             response = model.generate_content(prompt)
#             #print(str(st.session_state.room_data)+prefix + prompt)

#             # Extract room data from the response and add to room_data
#             new_rooms = convert_to_room_data(response.text)
#             st.session_state.room_data.extend(new_rooms)
            
#             # Add model response to chat history
#             st.session_state.messages.append({"role": "model", "parts": [response.text]})
        
            
#             with st.sidebar:
#                 st.write("Updated Room Data:")
#                 st.write(st.session_state.room_data)
            
#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         role = message["role"]
#         parts = message["parts"]
#         with st.chat_message(role):
#             for part in parts:
#                 st.write(part)

#     # Generate Layout Button
#     if st.button("Generate Layout"):
#         generate_layout(st.session_state.room_data)

#     # Clear Layout Button
#     if st.button("Clear Layout"):
#         st.session_state.room_data = []
#         st.success("Layout cleared!")

@st.cache_data()
def lottie_local(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
    
def main():
    st.set_page_config(page_title="Archi Bot",page_icon=":book:")
    st.title("Archi Bot")

    # About section
    with st.sidebar:
        anim = lottie_local('animation.json')
        st_lottie(anim,
                speed=1,
                reverse=False,
                loop=True,
                height = 130,
                width = 250,
                quality="high",
            key=None)
        
        st.markdown("# About")
        st.markdown("ARCHI BOT is your friendly assistant for layout generation and validation.")
        st.markdown("Feel free to chat with ARCHI to get started!")
        st.markdown("- ARCHI provides assistance in creating house layouts.")
        st.markdown("- Validate your layouts against standards and requirements such as TNCDBR 2019.")
        st.markdown("- Save time and effort in your design process.")

        # Clear Layout Button
        if st.button("Clear Layout"):
            st.session_state.room_data = []
            st.success("Layout cleared!")

    # Initialize session state for room data and messages
    if 'room_data' not in st.session_state:
        st.session_state.room_data = []
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # React to user input
    if prompt := st.chat_input("What's up"):
        
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "parts": [prompt]})

        # Determine if "validate" is present in the prompt
        if re.search(r'\bvalidate\b', prompt, re.IGNORECASE):
            prompt = validate_text + prompt + str(st.session_state.room_data)
        else:
            prompt = prefix + prompt + str(st.session_state.room_data)
        
        # Generate response from Gemini API
        if st.session_state.messages:
            response = model.generate_content(prompt)

            # Extract room data from the response and add to room_data
            new_rooms = convert_to_room_data(response.text)
            st.session_state.room_data.extend(new_rooms)
            
            # Add model response to chat history
            st.session_state.messages.append({"role": "model", "parts": [response.text]})
        
            # Display updated room data
            with st.sidebar:
                st.write("Updated Room Data:")
                st.write(st.session_state.room_data)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        role = message["role"]
        parts = message["parts"]
        with st.chat_message(role):
            for part in parts:
                st.write(part)
    
    # Generate Layout Button
    if st.button("Generate Layout"):
        generate_layout(st.session_state.room_data)

if __name__ == "__main__":
    main()
