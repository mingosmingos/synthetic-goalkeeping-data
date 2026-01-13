import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from physicsbasedposes import generate_pose

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
    
    # Convert pose to DataFrame (same logic as in physicsbasedposes.py)
    pose_df = pd.DataFrame(pose).T  # Transpose to have nodes as rows
    pose_df.columns = ['x', 'y']
    pose_df.index.name = 'node'
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Plot nodes with labels
    ax.scatter(pose_df['x'], pose_df['y'], s=100, c='blue', alpha=0.6, edgecolors='black')
    
    # Label each node
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
    
    # Plot shot location
    ax.scatter(shot_x, shot_y, s=200, marker='*', color='red', label='Shot', zorder=3)

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
    
    # Display pose data
    st.subheader("Pose Data")
    st.dataframe(pose_df)
    st.json(pose_data)