# ARCHITECT: AI-Powered Architectural Design Platform

ARCHITECT (Automated Responsive Creation of House Interior and Exterior Configurations and Transformations) is a modular Generative AI-based system designed to streamline architectural workflows and interior design processes. The platform combines NLP, computer vision, and deep learning to automate layout generation, design visualization, and compliance validation—empowering architects, designers, and homeowners with accessible, intelligent tools.

## Core Features

### 1. House Layout Generation via Prompts
- Converts natural language prompts into structured room graphs using a custom NER model.
- Employs Graph Convolutional Networks (GCNs) to capture room relationships and an MLP for predicting room bounding boxes.
- Complies with local housing regulations such as TNCDBR 2019.

### 2. Visual House Layout + Vastu Compliance
- Uses Graph Neural Networks (GAT, GraphSAGE, TAGCN) to:
  - Suggest room types from visual prompts.
  - Compute Vastu Shastra compliance scores.
- Processes room orientation and adjacency using geometric analysis for accurate modeling.

### 3. Interior Remodelling Visualization
- Applies Neural Style Transfer (NST) using VGG-19 to render stylistic transformations.
- Integrates object segmentation (Detectron2) to preserve furniture realism.
- Post-processed using OpenCV for inpainting and refinement.

### 4. Rapid Façade Generation
- Trains multiple CycleGAN and Asymmetric CycleGAN configurations on CMP façade dataset.
- Accepts designer sketches and auto-generates detailed façade renderings.
- Enhances output via unsharp filtering and gamma correction.

### 5. Floorplan Vectorization & 3D Visualization
- Converts 2D raster floorplans into vectorized layouts using a VGG16-based encoder-decoder architecture.
- Classifies room boundaries, types, and openings.
- Visualized via Blender Python API and a pygame-based lightweight viewer for GPU-free interaction.

## Tech Stack

| Component              | Technology                                              |
|------------------------|----------------------------------------------------------|
| Prompt Understanding   | Custom NER, Graph Construction                          |
| Graph Learning         | PyTorch Geometric, GCN, GAT, TAGCN, GraphSAGE          |
| Style Transfer         | VGG-19, NST, OpenCV, Detectron2                         |
| GAN Training           | CycleGAN, Asymmetric GANs                              |
| Floorplan Modeling     | VGG-16 Encoder, Conv2DTranspose Decoder, TFRecords     |
| 3D Visualization       | Blender bpy API, pygame                                 |
| Datasets               | Tell2Design, RPLAN, CMP, Custom Vastu Dataset          |

## Performance & Evaluation

- Prompt-driven layout generation achieves high IoU and layout compliance.
- GNN-based Vastu scoring demonstrates superior accuracy over MLP baselines.
- NST results achieve high SSIM and LPIPS similarity scores.
- Façade generation yields diverse and structurally consistent outputs post-asymmetric GAN optimization.
- Vectorization outputs maintain over 90% precision/recall on benchmarked test sets.


## Future Enhancements

- Integrate LLMs (e.g., LLaMA 3, GPT-4o) for end-to-end prompt-based layout + reasoning.
- Add VR walkthrough support using Unity/Unreal Engine.
- Enable multi-user collaboration and cloud deployment (e.g., via AWS/GCP).
- Expand Vastu validation to other cultural norms and global architectural guidelines.

## Citation & Credits

Developed by:
- Arunachalam M  
- Aswin Ramanathan V  
- Kishore Prashanth P  

Supervised by:  
Dr. Abirami Murugappan, Dept. of Information Science & Technology, CEG, Anna University.

Model weights for Vectorization of Floorplans (app folder)
https://drive.google.com/drive/folders/1RxZp8OxzcEvqzxriEphIltRlRyP7fD47?usp=sharing
