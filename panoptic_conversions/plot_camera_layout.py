import os
import json
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def cam_center(R, t):
    # Center = -R^T * t
    Cx = -(R[0][0]*t[0][0] + R[1][0]*t[1][0] + R[2][0]*t[2][0])
    Cy = -(R[0][1]*t[0][0] + R[1][1]*t[1][0] + R[2][1]*t[2][0])
    Cz = -(R[0][2]*t[0][0] + R[1][2]*t[1][0] + R[2][2]*t[2][0])
    return Cx, Cy, Cz


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--paramsdir', type=str, required=True, help='Directory containing camera calibration files')
    parser.add_argument('--axes', type=str, default='xz', choices=['xy', 'xz', 'yz'], help='Pair of axes to plot: xz (Top-down), xy (Front), yz (Side)')
    args = parser.parse_args()

    cameras = [f"calib_hd_00_{i:02d}.json" for i in range(1, 31)]    # All levels
    # cameras = [f"calib_hd_00_{i:02d}.json" for i in range(1, 11)]  # Lower level
    # cameras = [f"calib_hd_00_{i:02d}.json" for i in range(11, 21)] # Middle level
    # cameras = [f"calib_hd_00_{i:02d}.json" for i in range(21, 31)] # Upper level


    nodes_x = []
    nodes_y = []
    nodes_z = []
    labels = []
    camera_colors = []
    level_colors = ['#e74c3c', '#3498db', '#2ecc71']

    # Extract camera centers
    for f_name in cameras:
        file_path = os.path.join(args.paramsdir, f_name)
        
        if not os.path.exists(file_path):
            print(f"Warning: File {f_name} not found in {args.paramsdir}")
            continue

        with open(file_path, 'r') as f:
            cam = json.load(f)
            Cx, Cy, Cz = cam_center(cam["R"], cam["t"])

            nodes_x.append(Cx)
            nodes_y.append(Cy)
            nodes_z.append(Cz)
            
            name = cam['name'].replace('hd_00_', '')
            labels.append(name)

            node_id = cam['node']
            if 1 <= node_id <= 10:
                camera_colors.append(level_colors[0])
            elif 11 <= node_id <= 20:
                camera_colors.append(level_colors[1])
            else:
                camera_colors.append(level_colors[2])    
        
            print(f"Processed {name}: X={Cx:.2f}, Y={Cy:.2f}, Z={Cz:.2f}")

    if not nodes_x:
        print("No cameras were processed. Check filenames.")
        exit()

    # Data selection based on the --axes parameter
    if args.axes == 'xy':
        h_data, v_data = nodes_x, nodes_y
        h_label, v_label = 'X (mm)', 'Y (mm)'
        title = 'Front View'
    elif args.axes == 'xz':
        h_data, v_data = nodes_x, nodes_z
        h_label, v_label = 'X (mm)', 'Z (mm)'
        title = 'Zenithal View'
    else: # yz
        h_data, v_data = nodes_y, nodes_z
        h_label, v_label = 'Y (mm)', 'Z (mm)'
        title = 'Side View'

    # Create Zenithal Plot
    plt.figure(figsize=(10, 10))
    limit = 300
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    
    # Draw cameras
    plt.scatter(h_data, v_data, color=camera_colors, s=150, edgecolors='black', zorder=3)
    
    # Add labels
    for i, txt in enumerate(labels):
        plt.annotate(txt, (h_data[i], v_data[i]), xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    
    # Levels leyend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Upper level (21-30)', markerfacecolor=level_colors[2], markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Middle level (11-20)', markerfacecolor=level_colors[1], markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Lower level (01-10)', markerfacecolor=level_colors[0], markersize=12)
    ]
    plt.legend(handles=legend_elements, loc='upper right', title="Panels levels")
    
    # Dibujar un círculo de referencia de 5.5m
    # Draw reference circle
    circle = plt.Circle((0, 0), limit, color='green', fill=False, linestyle='--', alpha=0.3, label='Limit 5.5m')
    plt.gca().add_patch(circle)
    
    # Plot conf
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.title(f'{title} of Camera Distribution', fontsize=15)    
    plt.xlabel(h_label, fontsize=12)
    plt.ylabel(v_label, fontsize=12)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.show()