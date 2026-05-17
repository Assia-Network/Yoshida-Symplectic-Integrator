import os
import numpy as np
import pyvista as pv

# --- DATA ---
r = np.load(r"results/positions.npz")['pos'] 

body_names = [
    "Sun",
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
    "Pluto"
]

colors = [
    "#FFD700",  # Sun
    "#1E90FF",  # Mercury
    "#FF4500",  # Venus
    "#32CD32",  # Earth
    "#FF8C00",  # Mars
    "#8A2BE2",  # Jupiter
    "#00CED1",  # Saturn
    "#FF00FF",  # Uranus
    "#7FFF00",  # Neptune
    "#FFFFFF"   # Pluto
]

# --- PLOT ---
plotter = pv.Plotter()
plotter.background_color = "black"
step = 200

# Transpose array to shape: (bodies, coordinates, timesteps)
position_list_new = np.transpose(r, (1, 2, 0))

for i in range(r.shape[1]):
    # Orbit trajectory point cloud
    cloud = pv.PolyData(position_list_new[i, :, ::step].T)
    plotter.add_points(cloud, render_points_as_spheres=True, point_size=5, color=colors[i], label=body_names[i])

    # Last recorded position (current planet position)
    last_point = pv.PolyData(position_list_new[i, :, -1].reshape(1, 3))
    plotter.add_points(
        last_point,
        render_points_as_spheres=True,
        point_size=15,   
        color=colors[i]
    )

plotter.add_text("Solar System - 10-Body Model", position="upper_edge", font_size=14, color="white")  
plotter.add_legend(bcolor=None, size=(0.3, 0.3), loc="lower right")

# Ensure the output directory exists and define the screenshot path
screenshot_path = r"images/solar_system_3d.png"

# Render the interactive window and save the high-resolution image
plotter.show(screenshot=screenshot_path)