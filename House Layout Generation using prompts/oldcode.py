# def main():
#     st.title("ARCHITECT.AI")

#     # Initialize chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # React to user input
#     if prompt := st.chat_input("Type your prompt here:"):
#         # Display user message in chat message container
#         st.session_state.messages.append({"role": "user", "parts": [prompt]})
        
#         # Generate response from Gemini API
#         if st.session_state.messages:
#             # Send the prompt with prefix to Gemini
#             response = model.generate_content(prefix + prompt)
            
#             # Extract room data from the response and add to room_data
#             new_rooms = convert_to_room_data(response.text)
#             st.session_state.room_data.extend(new_rooms)
            
#             # Add model response to chat history
#             st.session_state.messages.append({"role": "model", "parts": [response.text]})
            
#             # Display model response in chat
#             st.write("Model Response:")
#             st.write(response.text)
            
#             # Display updated room data
#             st.write("Updated Room Data:")
#             st.write(st.session_state.room_data)
            
#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         role = message["role"]
#         parts = message["parts"]
#         with st.chat_message(role):
#             for part in parts:
#                 st.write(part)

# ############################Layout generation code####################################

# # Function to generate bounding box with direction
# def generate_bounding_box_with_direction(min_x, max_x, min_y, max_y, direction, width, height):
#     center_x = (max_x + min_x) / 2
#     center_y = (max_y + min_y) / 2
    
#     if direction == "North":
#         y1 = center_y
#         y2 = min(center_y + height, max_y)
#         x1 = max(center_x - width / 2, min_x)
#         x2 = min(center_x + width / 2, max_x)
#     elif direction == "North east":
#         y1 = center_y
#         y2 = min(center_y + height, max_y)
#         x1 = center_x
#         x2 = min(center_x + width, max_x)
#     elif direction == "East":
#         y1 = max(center_y - height / 2, min_y)
#         y2 = min(center_y + height / 2, max_y)
#         x1 = center_x
#         x2 = min(center_x + width, max_x)
#     elif direction == "South east":
#         y1 = max(center_y - height, min_y)
#         y2 = center_y
#         x1 = center_x
#         x2 = min(center_x + width, max_x)
#     elif direction == "South":
#         y1 = max(center_y - height, min_y)
#         y2 = center_y
#         x1 = max(center_x - width / 2, min_x)
#         x2 = min(center_x + width / 2, max_x)
#     elif direction == "South west":
#         y1 = max(center_y - height, min_y)
#         y2 = center_y
#         x1 = max(center_x - width, min_x)
#         x2 = center_x
#     elif direction == "West":
#         y1 = max(center_y - height / 2, min_y)
#         y2 = min(center_y + height / 2, max_y)
#         x1 = max(center_x - width, min_x)
#         x2 = center_x
#     elif direction == "North west":
#         y1 = center_y
#         y2 = min(center_y + height, max_y)
#         x1 = max(center_x - width, min_x)
#         x2 = center_x
    
#     return (x1, y1, x2, y2)

# # Function to generate non-overlapping bounding boxes
# def generate_non_overlapping_bounding_boxes(room_data, min_x, max_x, min_y, max_y, min_distance):
#     bounding_boxes = []
#     centers = []
    
#     for room in room_data:
#         room_name = room['name']
#         width = room['width']
#         height = room['height']
#         direction = room['direction']
#         color = room['color']
        
#         while True:
#             bounding_box = generate_bounding_box_with_direction(min_x, max_x, min_y, max_y, direction, width, height)
#             center_x = (bounding_box[0] + bounding_box[2]) / 2
#             center_y = (bounding_box[1] + bounding_box[3]) / 2
#             new_center = (center_x, center_y)
            
#             if not check_center_collision(new_center, centers, min_distance):
#                 bounding_boxes.append({'bbox': bounding_box, 'color': color, 'name': room_name})
#                 centers.append(new_center)
#                 break
    
#     return bounding_boxes

# # Function to check center collision
# def check_center_collision(new_center, existing_centers, min_distance):
#     for center in existing_centers:
#         distance = ((new_center[0] - center[0]) ** 2 + (new_center[1] - center[1]) ** 2) ** 0.5
#         if distance < min_distance:
#             return True
#     return False

# ###########################################################################

# def execute_prompt(prompt):
#     response = model.generate_content(prompt)
#     return response.text

# def main():
#     st.title("ARCHITECT.AI")

#     # Initialize chat history
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # React to user input
#     if prompt := st.chat_input("What is up?"):
#         # Display user message in chat message container
#         st.session_state.messages.append({"role": "user", "parts": [prompt]})
        
#         # Build a list of user messages and model responses for context
#         messages = []
#         for message in st.session_state.messages:
#             role = message["role"]
#             parts = message["parts"]
#             messages.append({"role": role, "parts": parts})

#         # Generate response from Gemini API
#         if messages:
#             response = model.generate_content(messages)

#             # Add model response to chat history
#             st.session_state.messages.append({"role": "model", "parts": [response.text]})

#             # Display model response in chat
#             # st.write(response.text)

#     # Display chat messages from history on app rerun
#     for message in st.session_state.messages:
#         role = message["role"]
#         parts = message["parts"]
#         # with st.chat_message(role):
#         #     for part in parts:
#         #         st.markdown(part)
#         with st.chat_message(role, avatar=None if role == "user" else model_avatar):
#             for part in parts:
#                 if role == "user":
#                     st.write(part)  # Display user input directly
#                 else:
#                     st.markdown(part) 


# if __name__ == "__main__":
#     main()






# Function to visualize bounding boxes and save as image
# def visualize_bounding_boxes(bounding_boxes, min_x, max_x, min_y, max_y):
#     # Set non-interactive backend
#     matplotlib.use('Agg')
    
#     fig, ax = plt.subplots()
#     room_colors = {}  # To store room colors for legend
    
#     for room in bounding_boxes:
#         bounding_box = room['bbox']
#         color = room['color']
#         room_name = room['name']
        
#         # Plot bounding box with specified color
#         x_min, y_min, x_max, y_max = bounding_box
#         width = x_max - x_min
#         height = y_max - y_min
#         rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor=color, facecolor=color, label=room_name)
#         ax.add_patch(rect)
        
#         # Add room name as annotation
#         center_x = (x_min + x_max) / 2
#         center_y = (y_min + y_max) / 2
        
#         # Store room colors for legend
#         room_colors[room_name] = color
    

#     ax.set_xlim(min_x, max_x)
#     ax.set_ylim(min_y, max_y)
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')

#     # to send to kp for postprocess module
#     postprocess_path = 'kp_layout.png'
#     plt.axis("off")
#     plt.savefig(postprocess_path, bbox_inches='tight', pad_inches=0)

#     # bring back axis and title
#     plt.axis("on")
#     ax.set_title('Generated House Layout')

#     # Add legend
#     handles = [patches.Patch(color=color, label=room_name) for room_name, color in room_colors.items()]
#     plt.legend(handles=handles, loc='upper right', title='Rooms', fontsize='small')
    
#     # plt.grid(True)
    
#     # Save the plot as an image file
#     img_path = 'layout.png'  # Path to save the image
#     plt.savefig(img_path)

    
#     # Close the plot to release resources
#     plt.close()
    
#     return img_path
