import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pandas as pd

from physicsbasedposes import generate_pose, pose_to_dataframe, evaluate_save

st.set_page_config(page_title="Goalkeeper Pose Generator", layout="wide")

st.title("Goalkeeper Pose Generator")

# Input section
col1, col2 = st.columns(2)

with col1:
    shot_x = st.slider("Shot X Coordinate", min_value=0.0, max_value=27.0, value=13.5)

with col2:
    shot_y = st.slider("Shot Y Coordinate", min_value=0.0, max_value=27.0, value=13.5)

# Generate button
if st.button("Generate Pose", key="generate"):
    shot_coordinates = [shot_x, shot_y]
    
    # Generate the pose
    pose = generate_pose(shot_coordinates)
    
    # Convert pose to DataFrame and evaluate save
    pose_df = pose_to_dataframe(pose)
    eval_result = evaluate_save(pose_df, shot_coordinates, radius=1.0)
    nearest = eval_result['nearest_node']
    dist = eval_result['distance']
    radius = eval_result['radius']
    saved = eval_result['saved']

    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 12))

    # Plot nodes with labels
    ax.scatter(pose_df['x'], pose_df['y'], s=100, c='blue', alpha=0.6, edgecolors='black')

    for node, (x, y) in pose_df.iterrows():
        ax.annotate(node, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Draw skeleton connections (same as in physicsbasedposes.py)
    connections = [
        ('torso', 'head'),
        ('torso', 'left_shoulder'), ('torso', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_hand'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_hand'),
        ('torso', 'left_hip'), ('torso', 'right_hip'),
        ('left_hip', 'left_knee'), ('left_knee', 'left_foot'),
        ('right_hip', 'right_knee'), ('right_knee', 'right_foot')
    ]

    for node1, node2 in connections:
        x_vals = [pose_df.loc[node1, 'x'], pose_df.loc[node2, 'x']]
        y_vals = [pose_df.loc[node1, 'y'], pose_df.loc[node2, 'y']]
        ax.plot(x_vals, y_vals, 'k-', alpha=0.5, linewidth=2)

    # Highlight nearest node and draw radius circle
    nx, ny = pose_df.loc[nearest, 'x'], pose_df.loc[nearest, 'y']
    ax.scatter(nx, ny, s=140, color='orange', zorder=4, label='Nearest node')
    circle = Circle((nx, ny), radius, color='green' if saved else 'red', fill=False, linestyle='--', linewidth=2, alpha=0.7)
    ax.add_patch(circle)

    # Plot shot location
    ax.scatter(shot_x, shot_y, s=200, marker='*', color='red', label='Shot', zorder=5)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Goalkeeper Pose Visualization')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.set_xlim(0, 27)
    ax.set_ylim(0, 27)
    
    # Display the plot
    st.pyplot(fig)

    # Save evaluation
    st.subheader("Save Evaluation")
    st.write(f"Nearest node: {nearest} — distance: {dist:.2f} — Saved: {'✅' if saved else '❌'}")
    
    # Display pose data
    st.subheader("Pose Data")
    st.dataframe(pose_df)