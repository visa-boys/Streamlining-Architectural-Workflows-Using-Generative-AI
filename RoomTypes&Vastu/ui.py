import tkinter as tk
from main import *
# from torch_geometric.nn import SAGEConv, GATConv, GCNConv, TAGConv
from torch_geometric.nn import TAGConv,SAGEConv
import torch
from model import Model
from visualize import visualize
import pickle
from orientation import *

class RectangleDrawer:
    def __init__(self, root):
        self.root = root
        self.canvas = tk.Canvas(root, width=256, height=256, bg="white")
        self.canvas.pack()
        self.rectangles = []
        
        self.canvas.bind("<Button-1>", self.start_rect)
        self.canvas.bind("<B1-Motion>", self.draw_rect)
        self.canvas.bind("<ButtonRelease-1>", self.end_rect)
        
        self.start_x = None
        self.start_y = None
        self.curr_rect = None
        
    def start_rect(self, event):
        self.start_x = event.x
        self.start_y = event.y
        self.curr_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="black")
        
    def draw_rect(self, event):
        self.canvas.coords(self.curr_rect, self.start_x, self.start_y, event.x, event.y)
        
    def end_rect(self, event):
        self.rectangles.append((self.start_x, self.start_y, event.x, event.y))
        self.start_x = None
        self.start_y = None

def main():
    root = tk.Tk()
    root.title("Room Types Suggestion")
    app = RectangleDrawer(root)
    root.mainloop()
    rectangles = [[x, 255 - y1, x2, 255 - y2] for x, y1, x2, y2 in app.rectangles]
    print("Coordinates of all rectangles:", app.rectangles)
    with open('rectangles.pkl', 'wb') as f:
        pickle.dump(app.rectangles, f)
    orientation=ret_comb()
    d=fn(rectangles,orientation)
    
    #loaded_model.eval()
    with torch.no_grad():
        x, edge_index = d.x, d.edge_index
        new_predictions,_ = loaded_model(x, edge_index)
    predicted_categories = new_predictions.argmax(dim=1)
    predicted_room_categories = [MOD_ROOM_CLASS[idx.item()] for idx in predicted_categories]
    print(rectangles,predicted_room_categories)
    visualize(rectangles,predicted_room_categories)
if __name__ == "__main__":
    loaded_model = Model(layer_type=SAGEConv, n_hidden=6)
    loaded_model.load_state_dict(torch.load('notebook//SAGE_merged_trained_model_layers6.pth',map_location=torch.device('cpu')))
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
    main()

# models = {
#     'mlp': Linear,
#     'gcn': GCNConv,
#     'gat': GATConv,
#     'sage': SAGEConv,
#     'tagcn': TAGConv,
# }
# model_type = 'tagcn'
# hidden_layers = 2
