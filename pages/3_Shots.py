import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loaders import load_shots
from physicsbasedposes import pose_to_dataframe

st.title("Shots")

shots_df = load_shots()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    # Filter by result
    result_filter = st.radio("Result", ["All", "Goals", "Saves"])
    
    # Filter by appearance
    appearances = sorted(shots_df['appearance'].unique())
    selected_appearance = st.selectbox("Appearance", ["All"] + appearances)

# Apply filters
filtered_shots = shots_df.copy()

if result_filter == "Goals":
    filtered_shots = filtered_shots[filtered_shots['isgoal'] == True]
elif result_filter == "Saves":
    filtered_shots = filtered_shots[filtered_shots['isgoal'] == False]

if selected_appearance != "All":
    filtered_shots = filtered_shots[filtered_shots['appearance'] == selected_appearance]

st.subheader(f"Shots ({len(filtered_shots)})")

# Select a shot to view details
if not filtered_shots.empty:
    shot_options = filtered_shots.index.tolist()
    
    # Check if a shot was selected from another page
    default_shot = st.session_state.get('selected_shot', shot_options[0])
    if default_shot not in shot_options:
        default_shot = shot_options[0]
    
    # Get index in list for default value
    try:
        default_idx = shot_options.index(default_shot)
    except ValueError:
        default_idx = 0
    
    selected_shot_idx = st.selectbox(
        "Select a shot to view details:", 
        shot_options,
        index=default_idx,
        format_func=lambda x: f"Shot #{x} - {'⚽ Goal' if filtered_shots.loc[x, 'isgoal'] else '🧤 Save'}"
    )
    
    # Clear the session state after using it
    if 'selected_shot' in st.session_state:
        del st.session_state['selected_shot']
    
    if selected_shot_idx is not None:
        shot = filtered_shots.loc[selected_shot_idx]
        
        st.divider()
        st.subheader("Shot Details")
        
        # Display shot info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Shot ID", selected_shot_idx)
        with col2:
            st.metric("Result", "⚽ Goal" if shot['isgoal'] else "🧤 Save")
        with col3:
            st.metric("Position", f"({shot['x']:.1f}, {shot['y']:.1f})")
        with col4:
            st.metric("Velocity", f"{shot['velocity']:.1f}")
        
        # Visualize goalkeeper pose
        st.subheader("Goalkeeper Pose Visualization")
        
        # Extract pose data from the shot row
        body_nodes = [
            'torso', 'head',
            'left_shoulder', 'left_elbow', 'left_hand',
            'left_hip', 'left_knee', 'left_foot',
            'right_shoulder', 'right_elbow', 'right_hand',
            'right_hip', 'right_knee', 'right_foot'
        ]
        
        # Build pose dict from shot columns
        pose = {}
        for node in body_nodes:
            pose[node] = {
                'x': shot[f'{node}_x'],
                'y': shot[f'{node}_y']
            }
        
        pose_df = pose_to_dataframe(pose)
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Plot nodes
        ax.scatter(pose_df['x'], pose_df['y'], s=100, c='blue', alpha=0.6, edgecolors='black')
        
        # Label each node
        for node, (x, y) in pose_df.iterrows():
            ax.annotate(node, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Draw skeleton connections
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
        ax.scatter(shot['x'], shot['y'], s=200, marker='*', 
                  color='green' if shot['isgoal'] else 'red', 
                  label='Shot', zorder=5)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f"Shot #{selected_shot_idx} - {'Goal' if shot['isgoal'] else 'Save'}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.set_xlim(0, 27)
        ax.set_ylim(0, 27)
        
        st.pyplot(fig)
        
        # Display full shot data
        with st.expander("View Full Shot Data"):
            st.json(shot.to_dict())
else:
    st.info("No shots match the current filters")
