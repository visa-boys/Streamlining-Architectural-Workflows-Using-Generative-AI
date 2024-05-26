from misc import main_process, result_orient
import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse
import subprocess
import pickle
from visualize import *
from annotated_text import annotated_text
from streamlit_lottie import st_lottie
import json
import webbrowser
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from orient_new import *

MOD_ROOM_CLASS = {0: "Living room", 
                1: "Master room",
                2: "Kitchen",
                3: "Bathroom",
                4: "Dining room",
                5: "Child room",
                6: "Study room",
                7: "Second room",
                8: "Guest room",
                9: "Balcony",
                10: "Entrance",
                11: "Storage",
                12: "Wall-in"}
MOD_ROOM_CLASS_NEW = {0: "Living Room/Kitchen/Dining room", 
                1: "Hall",
                2: "Bedroom",
                3: "Closet",
                4: "Bathroom/Washroom",
                5: "Balcony",
}
vastu_scores_new = {
    0: [7, 6, 5, 4, 3, 2, 1, 0, 7],
    1: [6, 4, 5, 3, 1, 0, 2, 7, 6],
    2: [3, 0, 2, 1, 6, 7, 5, 4, 3],
    3: [6, 4, 5, 3, 1, 0, 2, 7, 6],
    4: [4, 0, 3, 2, 5, 1, 6, 7, 4],
    5: [5, 6, 4, 3, 2, 1, 0, 7, 5]
}

vastu_scores = {
    0: [7, 6, 5, 4, 3, 2, 1, 0, 7],
    1: [3, 0, 2, 1, 6, 7, 5, 4, 3],
    2: [2, 0, 5, 7, 3, 1, 4, 6, 2],
    3: [4, 0, 3, 2, 5, 1, 6, 7, 4],
    4: [6, 5, 7, 4, 1, 0, 2, 3, 6],
    5: [5, 3, 4, 6, 1, 0, 2, 7, 5],
    6: [6, 5, 4, 3, 2, 1, 0, 7, 6],
    7: [6, 5, 4, 3, 2, 1, 0, 7, 6],
    8: [6, 4, 5, 3, 1, 0, 2, 7, 6],
    9: [5, 6, 4, 3, 2, 1, 0, 7, 5],
    10: [5, 7, 6, 3, 1, 0, 2, 4, 5],
    11: [6, 4, 5, 3, 1, 0, 2, 7, 6],
    12: [6, 5, 4, 3, 2, 1, 0, 7, 6],
    13: [5, 7, 6, 3, 1, 0, 2, 4, 5]
}
@st.cache_data()
def lottie_local(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)
def calculate_vastu_scores(labels, orientations):
    room_indices = [list(MOD_ROOM_CLASS.keys())[list(MOD_ROOM_CLASS.values()).index(label)] for label in labels]
    vastu_scores_list = [vastu_scores[index][orientation] for index, orientation in zip(room_indices, orientations)]
    average_vastu_score = sum(vastu_scores_list) / (7*len(vastu_scores_list))
    #print(sum(vastu_scores_list),7*len(vastu_scores_list))
    return vastu_scores_list,average_vastu_score
def calculate_vastu_scores_new(labels, orientations):
    room_indices = [list(MOD_ROOM_CLASS_NEW.keys())[list(MOD_ROOM_CLASS_NEW.values()).index(label)] for label in labels]
    vastu_scores_list = [vastu_scores_new[index][orientation] for index, orientation in zip(room_indices, orientations)]
    average_vastu_score = sum(vastu_scores_list) / (7*len(vastu_scores_list))
    #print(sum(vastu_scores_list),7*len(vastu_scores_list))
    return vastu_scores_list,average_vastu_score
def coord_to_blend(saved_rectangles):
    # Define the dimensions of the canvas
    width = 256
    height = 256
    # Create a numpy array of all 1s
    canvas = np.ones((height, width))
    # Define the boundary width
    boundary_width = 3
    # Create a figure without displaying axis ticks and labels
    fig, ax = plt.subplots()
    ax.imshow(canvas, cmap='binary', origin='upper', extent=[0, width, 0, height])
    ax.axis('off')
    # Plot the rectangles with boundaries
    for rect_coords in saved_rectangles:
        x1, y1, x2, y2 = rect_coords
        rect_width = x2 - x1 + 1
        rect_height = y2 - y1 + 1
        # Create a rectangle patch with boundary 0
        rect = patches.Rectangle((x1, y1), rect_width, rect_height, linewidth=boundary_width, edgecolor='black', facecolor='none')
        # Add the rectangle patch to the plot
        ax.add_patch(rect)
    # Convert the plot to a numpy array
    fig.canvas.draw()
    rectangle_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    rectangle_image = rectangle_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # Remove the plot to clear memory
    plt.close(fig)
    # Convert the image to a binary array
    rectangle_array = np.all(rectangle_image == [255, 255, 255], axis=-1).astype(float)
    # Invert the array so 1 represents the rectangles and 0 represents the background
    rectangle_array = 1 - rectangle_array
    # Save the numpy array to a file
    np.save('array_file.npy', rectangle_array)
    #print(rectangle_array.shape)
    subprocess.run(["python", "blender_code.py"])
    #print("############################################################")

# Function to process the image
def process_image(image_path, postproc, colorize):
    image = cv2.imread(image_path)
    args = argparse.Namespace(image=image_path, weight='E://Blender//vectorize//log files//log//store//G', loadmethod='log', postprocess=postproc, colorize=colorize, save=None)
    result = main_process(args)
    
    if postproc and not colorize:
        #print("How many times")
        result[result != 10] = 0
        result[result == 10] = 1
        np.save('array_file.npy', result)
        subprocess.run(["python", "blender_code.py"]) 
    return result

# Main Streamlit app
def call():
    st.set_page_config(page_title="ARCHITECT", page_icon="	:house:")
    st.header(":house_buildings: ARCHITECT.ai :robot_face: ")
    #st.title('ARCHITECT')
    st.sidebar.write("# ARCHITECT.ai")
    with st.sidebar:
        annotated_text(
        ("Automated","A"),
        (" Responsive","R"),
        (" Creation","C"),
        (" of"),
        (" House","H"),
        (" Interior","IT"),
        (" and"),
        (" Exterior","E"),
        (" Configurations","C"),
        (" and"),
        (" Transformations","T"),
    )

    # Create sidebar for tabs
    tab = st.sidebar.radio("Select your assistant", ("Vectorization of Floorplans", "Visual Prompting","House Layout Generator","Interior Designing"))

    if tab == "Vectorization of Floorplans":
        task_1()
    elif tab == "Visual Prompting":
        task_2()
    elif tab=="House Layout Generator":
        task_3()
    elif tab=="Interior Designing":
        task_4()
    
    with st.sidebar:
        anim = lottie_local('Animation - 1714129899563.json')
        st_lottie(anim,
                speed=1,
                reverse=False,
                loop=True,
                height = 130,
                width = 250,
                quality="high",
            key=None)
def task_4():
    #st.subheader("Interior Remodelling Visualization")
    # # Define the URL you want to open
    # url_to_open = "https://fyp-website-1-astlefxdqgoujeqbbr57er.streamlit.app/"
    # #url_to_open_new = "https://archi-bot.streamlit.app"

    # # Create two columns for the cards
    # #col1, col2 = st.columns(2)

    # # Content and button for the first card
    # #with col1:
    # st.write("Style your house interior")
    # if st.button("Start",key='button1'):
    #     webbrowser.open_new_tab(url_to_open)
    #  st.subheader("House Layout Generator")
    # Define the URL you want to open
    url_to_open = "https://fyp-website-1-astlefxdqgoujeqbbr57er.streamlit.app/"
    url_to_open_new = "http://192.168.137.57:8501/"

    # Create two columns for the cards
    col1, col2 = st.columns(2)

    # Content and button for the first card
    with col1:
        st.write("Interior Remodelling Visualization")
        if st.button("Start",key='button1'):
            webbrowser.open_new_tab(url_to_open)
     # Content and button for the second card
    with col2:
        st.write("Rapid Facade Visualization")
        if st.button("Start",key='button2'):
            webbrowser.open_new_tab(url_to_open_new)
            
def task_1():
    st.subheader("Vectorization of Floorplans")
    # File uploader for image
    uploaded_file = st.file_uploader("Upload a 2D raster floorplan", type=["jpg", "jpeg"])
    if uploaded_file is not None:
        image_path = 'uploaded_image.jpg'
        with open(image_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        # st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display uploaded image
        fig = plt.figure(figsize=(8, 4))  # Adjust the size as needed
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.xticks([])
        plt.yticks([])
        plt.title("2D Raster Floorplan")
        
        # Checkbox for postprocessing and colorization
        postproc = st.checkbox('Room Boundary Prediction', value=True)
        colorize = st.checkbox('Room Types Prediction', value=True)
        
        # Check if at least one checkbox is selected
        if not (postproc or colorize):
            st.warning('Please select at least one option.')

        # Button to trigger image processing if at least one checkbox is selected
        if (postproc or colorize) and st.button("Vectorize"):
            # Process image based on user options
            result = process_image(image_path, postproc, colorize)

            # Display processed image
            plt.subplot(1, 2, 2)
            plt.imshow(result)
            plt.xticks([])
            plt.yticks([])
            plt.title("Vectorized Floorplan")

            # Show the plot
            st.pyplot(fig)
            if postproc==True and colorize==True:
                saved_orientation,saved_labels=result_orient(result)
                saved_orientation=ret_comb(saved_orientation)
                print("Vada")
                print(saved_orientation,saved_labels)
                vastu_scores_list,average_vastu_score = calculate_vastu_scores_new(saved_labels, saved_orientation)
                st.markdown("#### Vastu Compliance Score: {:.2f}".format(average_vastu_score))

            if postproc==True and colorize==False:
                # Display download link for 3droom.blend
                st.markdown("#### 3D Visualization using Blender")
                st.markdown("Click below to download the .blend floorplan file:")
                with open('3droom.blend', 'rb') as f:
                    file_bytes = f.read()
                st.download_button(label="Download", data=file_bytes, file_name='3droom.blend')
    else:
        st.warning('Please upload an image.')
def task_2():
    st.subheader("Visual Prompting")
    # Add your code for the second task here
    if st.button("Configure Visual Prompts"):
        subprocess.run(["python", "ui.py"])
        with open('data.pkl', 'rb') as f:
            saved_data = pickle.load(f)
        saved_rectangles, saved_labels,saved_orientation = saved_data
        #print(saved_rectangles)
        print("Data:")
        print(saved_labels,saved_orientation)
        fig = visualize(saved_rectangles, saved_labels)
        st.pyplot(fig, use_container_width=True)
        vastu_scores_list,average_vastu_score = calculate_vastu_scores(saved_labels, saved_orientation)
        st.markdown("#### Vastu Compliance Score: {:.2f}".format(average_vastu_score))
        coord_to_blend(saved_rectangles)
        # Display download link for 3droom.blend
        st.markdown("#### 3D Visualization using Blender")
        st.markdown("Click below to download the .blend floorplan file:")
        with open('3droom.blend', 'rb') as f:
            file_bytes = f.read()
        st.download_button(label="Download", data=file_bytes, file_name='3droom.blend')
        #st.write(vastu_scores_list)
# Define a function to load and process the numpy file
def process_numpy_file(uploaded_file):
    #print(uploaded_file)
    if uploaded_file is not None:
        #st.write(uploaded_file.name)  # Print the name of the uploaded file
        if uploaded_file.name.endswith('.npz') or uploaded_file.name.endswith('.npy'):
            data = np.load(uploaded_file, allow_pickle=True)
            data = data ^ 1  # Example operation, modify as needed
            np.save('array_file.npy', data)
            subprocess.run(["python", "blender_code.py"])
            st.markdown("#### 3D Visualization using Blender")
            st.markdown("Click below to download the .blend floorplan file:")
            with open('3droom.blend', 'rb') as f:
                file_bytes = f.read()
            st.download_button(label="Download", data=file_bytes, file_name='3droom.blend')
        else:
            st.write("Unsupported file format. Please upload a .npz or .npy file.")

def task_3():
    st.subheader("House Layout Generator")
    # Define the URL you want to open
    url_to_open = "https://house-layout-generator.streamlit.app"
    url_to_open_new = "https://archi-bot.streamlit.app"

    # Create two columns for the cards
    col1, col2 = st.columns(2)

    # Content and button for the first card
    with col1:
        st.write("Generate House Layouts from prompts or predefined menu driven room specifications")
        if st.button("Start",key='button1'):
            webbrowser.open_new_tab(url_to_open)
            # Create a file uploader widget
        uploaded_file_ = st.file_uploader("Upload generated house layout npy file", type=["npy"])
        print(uploaded_file_)
        process_numpy_file(uploaded_file_) 

        # Create a file uploader widget
        uploaded_file_json = st.file_uploader("Upload generated house layout JSON file", type=["json"])

        if uploaded_file_json is not None:
            # Read and decode the JSON file
            json_data = uploaded_file_json.read().decode("utf-8")
            data = json.loads(json_data)

            direction_mapping = {
                "North": 0,
                "North east": 1,
                "East": 2,
                "South east": 3,
                "South": 4,
                "South west": 5,
                "West": 6,
                "North west": 7
            }

            # Extract room labels and direction indices
            saved_labels = [room["name"] for room in data]
            saved_orientation = [direction_mapping[room["direction"]] for room in data]

            # Print the extracted labels and orientations
            print(saved_labels)  # Output: ['Balcony', 'Kitchen']
            print(saved_orientation)  # Output: [0, 6]

            vastu_scores_list,average_vastu_score = calculate_vastu_scores(saved_labels, saved_orientation)
            st.markdown("#### Vastu Compliance Score: {:.2f}".format(average_vastu_score))



    # Content and button for the second card
    with col2:
        st.write("Generate House Layouts dynamically and validate layouts against building regulations.")
        if st.button("Start",key='button2'):
            webbrowser.open_new_tab(url_to_open_new)




if __name__ == "__main__":
    call()
