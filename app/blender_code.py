import bpy
import numpy as np
import io
from zipfile import ZipFile


# Create Blender mesh from numpy array
def create_mesh_from_array(array, height):
    verts = []
    faces = []

    # Loop through the array and create vertices and faces for walls
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] == 1:
                verts.append((i, j, 0))
                verts.append((i + 1, j, 0))
                verts.append((i + 1, j + 1, 0))
                verts.append((i, j + 1, 0))

                base_index = len(verts) - 4
                faces.append((base_index, base_index + 1, base_index + 2, base_index + 3))

                # Create top face for the wall
                verts.append((i, j, height))
                verts.append((i + 1, j, height))
                verts.append((i + 1, j + 1, height))
                verts.append((i, j + 1, height))

                base_index = len(verts) - 4
                faces.append((base_index + 3, base_index + 2, base_index + 1, base_index))

                # Connect lower and upper layers with vertical faces
                verts.append((i, j, 0))
                verts.append((i, j, height))
                verts.append((i + 1, j, 0))
                verts.append((i + 1, j, height))
                verts.append((i + 1, j + 1, 0))
                verts.append((i + 1, j + 1, height))
                verts.append((i, j + 1, 0))
                verts.append((i, j + 1, height))

                base_index = len(verts) - 8
                faces.extend([
                    (base_index, base_index + 1, base_index + 3, base_index + 2),
                    (base_index + 2, base_index + 3, base_index + 5, base_index + 4),
                    (base_index + 4, base_index + 5, base_index + 7, base_index + 6),
                    (base_index + 6, base_index + 7, base_index + 1, base_index),
                ])

    return verts, faces
def set_viewport_shading(shading_type):
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                for space in area.spaces:
                    if space.type == 'VIEW_3D':
                        space.shading.type = shading_type
def save_blender_file(filename):
    #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    bpy.ops.wm.save_as_mainfile(filepath=filename)

def make_3D():
    # Define parameters
    wall_height = 40.0  # Height of the walls
    ceiling_height = wall_height - 20  # Set ceiling height to match wall height
    image_path = r"E:\Blender\vectorize\array_file.npy"  # Path to your numpy array image

    # Load numpy array image
    image_data = np.load(image_path, allow_pickle=True)
    #image_data=result
    # Create mesh data for walls
    set_viewport_shading('SOLID')
    wall_verts, wall_faces = create_mesh_from_array(image_data, wall_height)
    #print("Hello1")

    # Shift vertices to center the house layout at the origin
    min_x = min(v[0] for v in wall_verts)
    min_y = min(v[1] for v in wall_verts)
    wall_verts = [(v[0] - min_x, v[1] - min_y, v[2]) for v in wall_verts]
    #print("Hello2")

    # Create wall mesh object
    wall_mesh = bpy.data.meshes.new(name="WallMesh")
    wall_mesh.from_pydata(wall_verts, [], wall_faces)
    wall_mesh.update()
    #print("Hello3")

    # Create wall object
    wall_obj = bpy.data.objects.new(name="WallObject", object_data=wall_mesh)
    bpy.context.collection.objects.link(wall_obj)
    #print("Hello4")

    # Create and assign wall material
    wall_mat = bpy.data.materials.new(name="WallMaterial")
    wall_mat.diffuse_color = (0.784, 0.902, 0.784, 1.0)  # Adjust RGB values as needed
    wall_obj.data.materials.append(wall_mat)
    #print("Hello5")

    # Set wall object location and select it
    wall_obj.location = (0, 0, 0)
    wall_obj.select_set(True)
    bpy.context.view_layer.objects.active = wall_obj
    #print("Hello6")

    # Calculate bounding box of rooms
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), -float('inf'), -float('inf')
    for v in wall_verts:
        min_x = min(min_x, v[0])
        max_x = max(max_x, v[0])
        min_y = min(min_y, v[1])
        max_y = max(max_y, v[1])
    #print("Hello7")

    # Create base mesh data within bounding box
    base_verts = [(min_x, min_y, 0), (max_x, min_y, 0), (max_x, max_y, 0), (min_x, max_y, 0)]
    base_faces = [(0, 1, 2, 3)]
    #print("Hello8")

    # Create base mesh object
    base_mesh = bpy.data.meshes.new(name="BaseMesh")
    base_mesh.from_pydata(base_verts, [], base_faces)
    base_mesh.update()
    #print("Hello9")

    # Create base object
    base_obj = bpy.data.objects.new(name="BaseObject", object_data=base_mesh)
    bpy.context.collection.objects.link(base_obj)
    #print("Hello10")

    # Create and assign floor material
    floor_mat = bpy.data.materials.new(name="FloorMaterial")
    floor_mat.diffuse_color = (0.737, 0.561, 0.561, 1.0)  # Adjust RGB values as needed
    floor_mat.use_backface_culling = False  # Disable backface culling
    base_obj.data.materials.append(floor_mat)
    #print("Hello11")

    # Set base object location and select it
    base_obj.location = (0, 0, -0.1)  # Place base slightly below the walls to avoid Z-fighting
    base_obj.select_set(True)
    bpy.context.view_layer.objects.active = base_obj
    #print("Hello12")

    # Create ceiling mesh data within bounding box
    ceiling_verts = [(min_x, min_y, ceiling_height), (max_x, min_y, ceiling_height), (max_x, max_y, ceiling_height), (min_x, max_y, ceiling_height)]
    ceiling_faces = [(0, 1, 2, 3)]
    #print("Hello13")

    # Create ceiling mesh object
    ceiling_mesh = bpy.data.meshes.new(name="CeilingMesh")
    ceiling_mesh.from_pydata(ceiling_verts, [], ceiling_faces)
    ceiling_mesh.update()
    #print("Hello14")

    # Create ceiling object
    ceiling_obj = bpy.data.objects.new(name="CeilingObject", object_data=ceiling_mesh)
    bpy.context.collection.objects.link(ceiling_obj)
    #print("Hello15")

    # Create and assign ceiling material
    ceiling_mat = bpy.data.materials.new(name="CeilingMaterial")
    ceiling_mat.diffuse_color = (0.8, 0.3, 0.1, 1.0)  # Adjust RGB values as needed
    ceiling_mat.use_backface_culling = False  # Disable backface culling
    ceiling_obj.data.materials.append(ceiling_mat)

    # Set ceiling object location and select it
    ceiling_obj.location = (0, 0, ceiling_height)  # Adjust ceiling position
    ceiling_obj.select_set(True)
    bpy.context.view_layer.objects.active = ceiling_obj

    # Create camera
    camera = bpy.data.cameras.new("Camera")
    camera_obj = bpy.data.objects.new("Camera", camera)
    bpy.context.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj

    # Set the camera position outside the house, facing the center
    camera_distance = max(max_x - min_x, max_y - min_y) * 2  # Adjust the distance as needed
    camera_obj.location = ((min_x + max_x) / 2, (min_y + max_y) / 2 - camera_distance, wall_height)
    camera_obj.rotation_euler = (np.pi / 6, 0, 0)  # Adjust the rotation as needed
    #print("Hello16")

    # Set viewport shading
    #print("##################################",bpy.context.screen.areas)
    # for area in bpy.context.screen.areas:
    #     if area.type == 'VIEW_3D':
    #         for space in area.spaces:
    #             if space.type == 'VIEW_3D':
    #                 space.shading.type = 'SOLID'
    #print("Hello17")

    # Display result in viewport
    bpy.ops.object.mode_set(mode='OBJECT')
    #print("Hello18")
    bpy.ops.object.mode_set(mode='EDIT')
    #print("Hello19")
    bpy.ops.mesh.select_all(action='SELECT')
    #print("Hello20")
    #bpy.ops.mesh.normals_make_consistent(inside=False)
    #print("Hello21")
    bpy.ops.object.mode_set(mode='OBJECT')
    #print("Hello22")

    save_blender_file("3droom.blend")
    # blend_io = io.BytesIO()
    # bpy.ops.wm.save_as_mainfile(filepath=blend_io, check_existing=False)

    # return blend_io
make_3D()


