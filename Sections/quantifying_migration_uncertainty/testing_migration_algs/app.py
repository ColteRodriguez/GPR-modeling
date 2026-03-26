import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
import numpy as np
from scipy.ndimage import label, center_of_mass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import center_of_mass

class VelocityModelBuilder:
    def __init__(self, migrated_image, extent=None, callback=None):
        """
        Interactive tool for drawing velocity structures on migrated GPR section
        
        Parameters:
        -----------
        migrated_image : 2D array
            The migrated GPR section to interpret
        extent : list
            [xmin, xmax, zmin, zmax] for axis labels
        callback : function, optional
            Function to call with (velocity_model, objects) when export is clicked
        """
        self.migrated_image = migrated_image
        self.nz, nx = migrated_image.shape
        self.extent = extent or [0, nx, self.nz, 0]
        self.callback = callback
        
        # Storage for drawn objects
        self.objects = []
        self.velocity_model = np.zeros_like(migrated_image, dtype=int)
        
        # Drawing state
        self.current_tool = 'polygon'
        self.current_points = []
        self.is_drawing = False
        self.temp_line = None
        self.temp_patches = []
        
        # Setup GUI
        self.setup_gui()
        
    def setup_gui(self):
        """Create the main window and widgets"""
        self.root = tk.Tk()
        self.root.title("GPR Velocity Model Builder")
        
        # Main layout: left side = image, right side = controls
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - matplotlib figure
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky='nsew')
        
        self.fig = Figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=left_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Display migrated image
        im = self.ax.imshow(self.migrated_image, aspect='auto', cmap='seismic',
                           extent=self.extent, interpolation='bilinear')
        self.ax.set_xlabel('Distance (m)')
        self.ax.set_ylabel('Depth (m)')
        self.ax.set_title('Click to draw structures (press Enter when done with each object)')
        self.fig.colorbar(im, ax=self.ax, label='RTM Amplitude')
        self.canvas.draw()
        
        # Connect mouse events
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.root.bind('<Return>', self.finish_object)
        self.root.bind('<Escape>', self.cancel_drawing)
        
        # Right panel - controls
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky='nsew', padx=(10, 0))
        
        # Tool selection
        ttk.Label(right_frame, text="Drawing Tool:", font=('Arial', 12, 'bold')).pack(pady=5)
        
        self.tool_var = tk.StringVar(value='polygon')
        ttk.Radiobutton(right_frame, text="Polygon (freeform)", 
                       variable=self.tool_var, value='polygon',
                       command=self.change_tool).pack(anchor='w', padx=20)
        ttk.Radiobutton(right_frame, text="Rectangle", 
                       variable=self.tool_var, value='rectangle',
                       command=self.change_tool).pack(anchor='w', padx=20)
        
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Object ID input
        ttk.Label(right_frame, text="Object ID (velocity zone):", 
                 font=('Arial', 11, 'bold')).pack(pady=5)
        self.id_entry = ttk.Entry(right_frame, width=15, font=('Arial', 11))
        self.id_entry.pack(pady=5)
        self.id_entry.insert(0, "1")
        
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Instructions
        ttk.Label(right_frame, text="Instructions:", 
                 font=('Arial', 11, 'bold')).pack(pady=5)
        
        instructions = tk.Text(right_frame, height=12, width=30, wrap=tk.WORD, 
                              font=('Arial', 9), bg='#f0f0f0')
        instructions.pack(pady=5, padx=10)
        instructions.insert('1.0', 
            "POLYGON MODE:\n"
            "• Click to add points\n"
            "• Press ENTER to finish\n"
            "• Press ESC to cancel\n\n"
            "RECTANGLE MODE:\n"
            "• Click and drag\n"
            "• Release to finish\n\n"
            "After each object:\n"
            "• Enter Object ID\n"
            "• Press ENTER to save\n\n"
            "TIP: Use different IDs\n"
            "for different materials"
        )
        instructions.config(state='disabled')
        
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Object list
        ttk.Label(right_frame, text="Drawn Objects:", 
                 font=('Arial', 11, 'bold')).pack(pady=5)
        
        self.object_listbox = tk.Listbox(right_frame, height=8, font=('Arial', 9))
        self.object_listbox.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        ttk.Button(right_frame, text="Delete Selected", 
                  command=self.delete_object).pack(pady=5)
        
        ttk.Separator(right_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Action buttons
        button_frame = ttk.Frame(right_frame)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Clear All", 
                  command=self.clear_all).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Export Model", 
                  command=self.export_model).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Select a tool and start drawing")
        status_label = ttk.Label(self.root, textvariable=self.status_var, 
                                relief=tk.SUNKEN, anchor='w')
        status_label.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Configure grid weights
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
    def change_tool(self):
        """Handle tool changes"""
        self.current_tool = self.tool_var.get()
        self.cancel_drawing(None)
        self.status_var.set(f"Tool: {self.current_tool.upper()} - Start drawing")
        
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return
            
        if self.current_tool == 'polygon':
            self.current_points.append([event.xdata, event.ydata])
            self.is_drawing = True
            
            # Draw point
            self.ax.plot(event.xdata, event.ydata, 'ro', markersize=5)
            self.canvas.draw()
            
            self.status_var.set(f"Polygon: {len(self.current_points)} points - Press ENTER to finish")
            
        elif self.current_tool == 'rectangle':
            if not self.is_drawing:
                # Start rectangle
                self.rect_start = [event.xdata, event.ydata]
                self.is_drawing = True
                self.status_var.set("Rectangle: Drag to set size, release to finish")
            else:
                # Finish rectangle
                self.rect_end = [event.xdata, event.ydata]
                self.finalize_rectangle()
                
    def on_move(self, event):
        """Handle mouse movement (for rectangle preview)"""
        if not self.is_drawing or self.current_tool != 'rectangle':
            return
            
        if event.inaxes != self.ax:
            return
            
        # Clear previous preview
        if self.temp_line:
            self.temp_line.remove()
            
        # Draw preview rectangle
        x0, y0 = self.rect_start
        width = event.xdata - x0
        height = event.ydata - y0
        
        rect = patches.Rectangle((x0, y0), width, height, 
                                 linewidth=2, edgecolor='red', 
                                 facecolor='red', alpha=0.3)
        self.temp_line = self.ax.add_patch(rect)
        self.canvas.draw()
        
    def finalize_rectangle(self):
        """Complete rectangle drawing"""
        # Clear preview
        if self.temp_line:
            self.temp_line.remove()
            self.temp_line = None
            
        # Create rectangle coordinates (4 corners)
        x0, y0 = self.rect_start
        x1, y1 = self.rect_end
        
        self.current_points = [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1]
        ]
        
        # Draw final rectangle
        coords = np.array(self.current_points + [self.current_points[0]])
        self.ax.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=2)
        self.canvas.draw()
        
        self.is_drawing = False
        self.prompt_for_id()
        
    def finish_object(self, event):
        """Finish drawing current object (Enter key)"""
        if not self.current_points:
            return
            
        if self.current_tool == 'polygon':
            # Close the polygon
            coords = np.array(self.current_points + [self.current_points[0]])
            self.ax.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=2)
            self.canvas.draw()
            
            self.is_drawing = False
            self.prompt_for_id()
            
    def prompt_for_id(self):
        """Show dialog to get object ID"""
        try:
            obj_id = int(self.id_entry.get())
        except ValueError:
            messagebox.showerror("Invalid ID", "Please enter a valid integer for Object ID")
            self.cancel_drawing(None)
            return
            
        # Save object
        self.objects.append({
            'type': self.current_tool,
            'coords': np.array(self.current_points),
            'id': obj_id
        })
        
        # Update mask
        self.update_velocity_mask(self.current_points, obj_id)
        
        # Update listbox
        self.object_listbox.insert(tk.END, 
            f"Object {len(self.objects)}: ID={obj_id}, Type={self.current_tool}, Points={len(self.current_points)}")
        
        # Increment ID for next object
        self.id_entry.delete(0, tk.END)
        self.id_entry.insert(0, str(obj_id + 1))
        
        # Reset for next object
        self.current_points = []
        self.status_var.set(f"Object saved! Draw next object or Export Model")
        
    def update_velocity_mask(self, coords, obj_id):
        """Fill polygon region in velocity model mask"""
        from matplotlib.path import Path
        
        # Convert coordinates to pixel indices
        x_coords = np.array([c[0] for c in coords])
        y_coords = np.array([c[1] for c in coords])
        
        # Map from extent to array indices
        x_min, x_max, z_max, z_min = self.extent
        x_pixels = ((x_coords - x_min) / (x_max - x_min) * self.migrated_image.shape[1]).astype(int)
        y_pixels = ((y_coords - z_min) / (z_max - z_min) * self.migrated_image.shape[0]).astype(int)
        
        # Create path and test all pixels
        path = Path(list(zip(x_pixels, y_pixels)))
        
        y_grid, x_grid = np.mgrid[:self.nz, :self.migrated_image.shape[1]]
        points = np.vstack((x_grid.flatten(), y_grid.flatten())).T
        
        mask = path.contains_points(points).reshape(self.nz, -1)
        self.velocity_model[mask] = obj_id
        
    def cancel_drawing(self, event):
        """Cancel current drawing (Escape key)"""
        # Clear temp graphics
        if self.temp_line:
            self.temp_line.remove()
            self.temp_line = None
            
        for patch in self.temp_patches:
            patch.remove()
        self.temp_patches = []
        
        self.current_points = []
        self.is_drawing = False
        self.canvas.draw()
        self.status_var.set("Drawing cancelled - Start new object")
        
    def delete_object(self):
        """Delete selected object from list"""
        selection = self.object_listbox.curselection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select an object to delete")
            return
            
        idx = selection[0]
        obj = self.objects[idx]
        
        # Remove from mask
        self.velocity_model[self.velocity_model == obj['id']] = 0
        
        # Remove from list
        del self.objects[idx]
        self.object_listbox.delete(idx)
        
        # Redraw
        self.redraw_all()
        self.status_var.set("Object deleted")
        
    def clear_all(self):
        """Clear all objects"""
        if not messagebox.askyesno("Clear All", "Delete all drawn objects?"):
            return
            
        self.objects = []
        self.velocity_model = np.zeros_like(self.migrated_image, dtype=int)
        self.object_listbox.delete(0, tk.END)
        self.redraw_all()
        self.status_var.set("All objects cleared")
        
    def redraw_all(self):
        """Redraw the figure with all objects"""
        self.ax.clear()
        self.ax.imshow(self.migrated_image, aspect='auto', cmap='seismic',
                      extent=self.extent, interpolation='bilinear')
        self.ax.set_xlabel('Distance (m)')
        self.ax.set_ylabel('Depth (m)')
        self.ax.set_title('Click to draw structures (press Enter when done with each object)')
        
        # Redraw all saved objects
        for obj in self.objects:
            coords = np.array(list(obj['coords']) + [obj['coords'][0]])
            self.ax.plot(coords[:, 0], coords[:, 1], 'r-', linewidth=2)
            
        self.canvas.draw()
            
    def export_model(self):
        """Export the velocity model mask"""
        if len(self.objects) == 0:
            messagebox.showwarning("No Objects", "Draw some objects first!")
            return
            
        # Close the GUI and return
        self.root.quit()
        self.root.destroy()
        
    def run(self):
        """Start the GUI"""
        self.root.mainloop()
        return self.velocity_model, self.objects


# Usage example:
def convert_mask_to_velocity(velocity_mask, id_to_velocity_map):
    """
    Convert object ID mask to actual velocity values
    
    Parameters:
    -----------
    velocity_mask : 2D array of ints
        Mask where each pixel has an object ID
    id_to_velocity_map : dict
        Mapping from object ID to velocity (m/s)
        e.g., {0: 1.5e8, 1: 1.3e8, 2: 1.7e8}
    
    Returns:
    --------
    velocity_model : 2D array of floats
        Velocity at each pixel
    """
    velocity_model = np.zeros_like(velocity_mask, dtype=float)
    for obj_id, velocity in id_to_velocity_map.items():
        velocity_model[velocity_mask == obj_id] = velocity
    return velocity_model

def map_mask_to_template(mask, template):
    """
    Map mask labels to template labels based on spatial correspondence.
    
    Creates an output array with the same shape as template, where each unique 
    value in mask is mapped to the spatially-corresponding unique value in template.
    
    Parameters:
    -----------
    mask : ndarray
        Input array with labeled regions (can be different shape than template)
    template : ndarray
        Target array with labeled regions to approximate
        
    Returns:
    --------
    mapped : ndarray
        Array with same shape as template, same unique values as template,
        but structure approximated from mask
        
    Example:
    --------
    # Mask has 3 regions (IDs: 0, 1, 2) in different positions
    # Template has 3 regions (IDs: 0, 5, 10) 
    # Output will have IDs 0, 5, 10 positioned like mask's 0, 1, 2
    """
    
    # Get unique labels (excluding background if 0)
    mask_labels = np.unique(mask)
    template_labels = np.unique(template)
    
    if len(mask_labels) != len(template_labels):
        raise ValueError(f"mask and template must have same number of unique values. "
                        f"mask has {len(mask_labels)}, template has {len(template_labels)}")
    
    # Resize mask to match template shape
    from scipy.ndimage import zoom
    zoom_factors = np.array(template.shape) / np.array(mask.shape)
    mask_resized = zoom(mask, zoom_factors, order=0)  # nearest neighbor to preserve labels
    
    # Find centers of mass for each label in both arrays
    mask_centers = {}
    for label in mask_labels:
        center = center_of_mass(mask_resized == label)
        mask_centers[label] = np.array(center)
    
    template_centers = {}
    for label in template_labels:
        center = center_of_mass(template == label)
        template_centers[label] = np.array(center)
    
    # Create cost matrix: distance between each mask region and each template region
    cost_matrix = np.zeros((len(mask_labels), len(template_labels)))
    
    for i, mask_label in enumerate(mask_labels):
        for j, template_label in enumerate(template_labels):
            # Euclidean distance between centers
            distance = np.linalg.norm(mask_centers[mask_label] - template_centers[template_label])
            cost_matrix[i, j] = distance
    
    # Solve assignment problem: which mask label corresponds to which template label?
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Create mapping dictionary
    label_mapping = {}
    for mask_idx, template_idx in zip(row_ind, col_ind):
        mask_label = mask_labels[mask_idx]
        template_label = template_labels[template_idx]
        label_mapping[mask_label] = template_label
    
    # Apply mapping to resized mask
    mapped = np.zeros_like(template)
    for mask_label, template_label in label_mapping.items():
        mapped[mask_resized == mask_label] = mask_label
    
    return mapped