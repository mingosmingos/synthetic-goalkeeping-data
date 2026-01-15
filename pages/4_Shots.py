import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.loaders import load_shots, load_appearances, load_players
from physicsbasedposes import find_nearest_node, generate_pose, pose_to_dataframe

st.title("Shots")

shots_df = load_shots()
appearances_df = load_appearances()
players_df = load_players()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    
    # Filter by result
    result_filter = st.radio("Result", ["All", "Goals", "Saves"])
    
    # Filter by appearance
    appearances = sorted(shots_df['appearance_id'].unique())
    selected_appearance = st.selectbox("Appearance", ["All"] + appearances)

# Apply filters
filtered_shots = shots_df.copy()

if result_filter == "Goals":
    filtered_shots = filtered_shots[filtered_shots['saved'] == False]
elif result_filter == "Saves":
    filtered_shots = filtered_shots[filtered_shots['saved'] == True]

if selected_appearance != "All":
    filtered_shots = filtered_shots[filtered_shots['appearance_id'] == selected_appearance]

st.subheader(f"Shots ({len(filtered_shots)})")

# Select a shot to view details
if not filtered_shots.empty:
    shot_options = filtered_shots.index.tolist()

    # Seed the selectbox from navigation state once, then rely on widget state.
    nav_shot = st.session_state.pop('selected_shot', None)
    if nav_shot in shot_options:
        st.session_state['shot_selector'] = nav_shot
    if 'shot_selector' not in st.session_state or st.session_state['shot_selector'] not in shot_options:
        st.session_state['shot_selector'] = shot_options[0]

    selected_shot_idx = st.selectbox(
        "Select a shot to view details:",
        shot_options,
        key='shot_selector',
        format_func=lambda x: f"Shot #{x} - {'⚽ Goal' if not filtered_shots.loc[x, 'saved'] else '🧤 Save'}",
    )
    
    if selected_shot_idx is not None:
        shot = filtered_shots.loc[selected_shot_idx]

        # Resolve keeper name via appearance_id -> player_id -> player name
        keeper_name = None
        player_id = None
        appearance_id = shot.get('appearance_id')

        if appearance_id is not None and appearances_df is not None and not appearances_df.empty:
            if 'appearance_id' in appearances_df.columns:
                appearance_row = appearances_df[appearances_df['appearance_id'] == appearance_id]
            elif appearance_id in appearances_df.index:
                appearance_row = appearances_df.loc[[appearance_id]]
            else:
                appearance_row = pd.DataFrame()

            if not appearance_row.empty and 'player_id' in appearance_row.columns:
                player_id = appearance_row.iloc[0]['player_id']

        if player_id is not None and players_df is not None and not players_df.empty:
            if 'player_id' in players_df.columns:
                player_row = players_df[players_df['player_id'] == player_id]
            elif player_id in players_df.index:
                player_row = players_df.loc[[player_id]]
            else:
                player_row = pd.DataFrame()

            if not player_row.empty:
                for name_col in ('name', 'player_name', 'keeper_name'):
                    if name_col in player_row.columns:
                        keeper_name = str(player_row.iloc[0][name_col])
                        break

        if keeper_name is None:
            keeper_name = str(player_id) if player_id is not None else "Unknown"
        
        st.divider()
        st.subheader("Shot Details")
        
        # Display shot info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Keeper", keeper_name)
        with col2:
            st.metric("Result", "⚽ Goal" if not shot['saved'] else "🧤 Save")
        with col3:
            st.metric("Position", f"({shot['x']:.1f}, {shot['y']:.1f})")
        with col4:
            st.metric("Velocity", f"{shot['velocity']:.1f}")
        
        # Visualize goalkeeper pose
        st.subheader("Goalkeeper Pose Visualization")
        show_optimal_overlay = st.checkbox("Overlay optimal pose", value=True, key="overlay_optimal_pose")
        
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
        
        recorded_pose_df = pose_to_dataframe(pose)

        # Compute a "generated (optimal)" pose for the same shot + keeper
        generated_pose_df = None
        try:
            pid_for_opt = int(player_id) if player_id is not None and pd.notna(player_id) else None
            if pid_for_opt is not None:
                opt_pose = generate_pose(pid_for_opt, [float(shot["x"]), float(shot["y"])], float(shot["velocity"]))
                generated_pose_df = pose_to_dataframe(opt_pose)
        except Exception:
            generated_pose_df = None

        # Recorded pose is always the primary pose.
        main_pose_df = recorded_pose_df
        overlay_pose_df = generated_pose_df
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # If we can't generate the optimal pose, just hide the overlay.
        if overlay_pose_df is None or overlay_pose_df.empty:
            if show_optimal_overlay:
                st.caption("Optimal pose not available for this shot.")
            show_optimal_overlay = False

        # Plot nodes (primary) — recorded pose should never change transparency
        ax.scatter(main_pose_df['x'], main_pose_df['y'], s=100, c='blue', alpha=0.65, edgecolors='black')
        # Plot nodes (overlay) — optimal pose is always transparent when enabled
        if show_optimal_overlay and overlay_pose_df is not None and not overlay_pose_df.empty:
            ax.scatter(
                overlay_pose_df['x'],
                overlay_pose_df['y'],
                s=90,
                c='gray',
                alpha=0.16,
                edgecolors='none',
                label='Optimal pose (overlay)',
            )
        
        # Label each node
        for node, (x, y) in main_pose_df.iterrows():
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
            x_vals = [main_pose_df.loc[node1, 'x'], main_pose_df.loc[node2, 'x']]
            y_vals = [main_pose_df.loc[node1, 'y'], main_pose_df.loc[node2, 'y']]
            ax.plot(x_vals, y_vals, 'k-', alpha=0.5, linewidth=2)
            if show_optimal_overlay and overlay_pose_df is not None and node1 in overlay_pose_df.index and node2 in overlay_pose_df.index:
                ox_vals = [overlay_pose_df.loc[node1, 'x'], overlay_pose_df.loc[node2, 'x']]
                oy_vals = [overlay_pose_df.loc[node1, 'y'], overlay_pose_df.loc[node2, 'y']]
                ax.plot(ox_vals, oy_vals, color='gray', alpha=0.16, linewidth=2)

        # Highlight nearest node for the recorded (primary) pose
        radius = float(shot.get('radius', 1.0))
        nearest_node = None
        distance = None
        candidate = shot.get('nearest_node')
        if isinstance(candidate, str) and candidate in main_pose_df.index:
            nearest_node = candidate
            try:
                distance = float(shot.get('distance')) if pd.notna(shot.get('distance')) else None
            except Exception:
                distance = None
        else:
            try:
                nearest_node, distance = find_nearest_node(main_pose_df, [float(shot['x']), float(shot['y'])])
            except Exception:
                nearest_node, distance = None, None

        if isinstance(nearest_node, str) and nearest_node in main_pose_df.index:
            nx, ny = float(main_pose_df.loc[nearest_node, 'x']), float(main_pose_df.loc[nearest_node, 'y'])
            is_saved = bool(distance <= radius) if distance is not None else bool(shot.get('saved', False))
            ax.scatter(nx, ny, s=160, color='orange', zorder=4, label=f'Nearest node ({nearest_node})')
            circle = Circle(
                (nx, ny),
                radius,
                color='green' if is_saved else 'red',
                fill=False,
                linestyle='--',
                linewidth=2,
                alpha=0.7,
            )
            ax.add_patch(circle)
        
        # Plot shot location
        ax.scatter(shot['x'], shot['y'], s=200, marker='x', 
                  color='green' if shot['saved'] else 'red', 
                  label='Shot', zorder=5)
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        #ax.set_title(f"Shot #{selected_shot_idx} - {'Goal' if not shot['saved'] else 'Save'}")
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
