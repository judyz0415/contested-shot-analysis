import sys
import os
# sys.path.append(os.path.abspath('C:\\Users\\willz\\OneDrive\\Documents\\code\\sportscode'))

import glob
import json
import numpy as np
import pandas as pd
from math import sqrt
from scipy.interpolate import CubicSpline
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import HTML
# import utils
import importlib
plt.rcParams['animation.embed_limit'] = 2**128
# importlib.reload(utils)



####################################################################################
######################## VISUALIZATION #############################################
####################################################################################
# limb_names = {('l_shoulder', 'l_elbow'): 'l_upperarm', ('r_shoulder', 'r_elbow'): 'r_upperarm', 
#               ('l_elbow', 'l_wrist'): 'l_forearm', ('r_elbow', 'r_wrist'): 'r_forearm', 
#               ('l_shoulder','neck'): 'l_collar', ('r_shoulder','neck'): 'r_collar',
#               ('l_hip', 'midhip'): 'l_hipline', ('r_hip', 'midhip'): 'r_hipline',
#               ('l_hip','l_knee'): 'l_thigh',  ('r_hip','r_knee'): 'r_thigh',
#               ('l_knee','l_ankle'): 'l_shin', ('r_knee','r_ankle'): 'r_shin',
#               ('l_ankle','l_heel'): 'l_achilles', ('r_ankle','r_heel'): 'r_achilles',
#               ('l_heel','l_bigtoe'): 'l_innerfoot', ('r_heel','r_bigtoe'): 'r_innerfoot',
#               ('l_heel','l_smalltoe'): 'l_outerfoot', ('r_heel','r_smalltoe'): 'r_outerfoot',
#               ('l_smalltoe','l_bigtoe'): 'l_toeline', ('r_smalltoe','r_bigtoe'): 'r_toeline',
#               ('nose', 'neck'): 'throat', ('nose', 'r_ear'):'r_face', ('nose', 'l_ear'):'l_face',
#               ('l_eye', 'r_eye'): 'eyeline', ('l_ear', 'r_ear'): 'earline',
#               ('l_shoulder', 'l_hip'): 'l_lat', ('r_shoulder', 'r_hip'): 'r_lat'   
#               }


limb_names = {
    'throat': ('nose', 'neck'),
    'l_upperarm': ('lShoulder', 'lElbow'),
    'r_upperarm': ('rShoulder', 'rElbow'),
    'l_forearm': ('lElbow', 'lWrist'),
    'r_forearm': ('rElbow', 'rWrist'),
    'l_thumb_len': ('lWrist', 'lThumb'),
    'r_thumb_len': ('rWrist', 'rThumb'),
    'l_pinky_len': ('lWrist', 'lPinky'),
    'r_pinky_len': ('rWrist', 'rPinky'),
    'l_hand_width': ('lThumb', 'lPinky'),
    'r_hand_width': ('rThumb', 'rPinky'),
    'l_collar': ('neck', 'lShoulder'),
    'r_collar': ('rShoulder', 'neck'),
    'l_lat': ('lShoulder', 'lHip'),
    'r_lat': ('rHip', 'rShoulder'),
    'l_hipline': ('lHip', 'midHip'),
    'r_hipline': ('midHip', 'rHip'),
    'l_thigh': ('lHip', 'lKnee'),
    'r_thigh': ('rHip', 'rKnee'),
    'l_shin': ('lKnee', 'lAnkle'),
    'r_shin': ('rKnee', 'rAnkle'),
    'l_achilles': ('lAnkle', 'lHeel'),
    'r_achilles': ('rAnkle', 'rHeel'),
    'l_innerfoot': ('lHeel', 'lBigToe'),
    'r_innerfoot': ('rHeel', 'rBigToe'),
    'l_outerfoot': ('lHeel', 'lSmallToe'),
    'r_outerfoot': ('rHeel', 'rSmallToe'),
    'l_foot_width': ('lBigToe', 'lSmallToe'),
    'r_foot_width': ('rBigToe', 'rSmallToe')
}

# limb_names = {  ('nose', 'neck'): 'throat',
#                 ('l_shoulder', 'l_elbow'): 'l_upperarm', ('r_shoulder', 'r_elbow'): 'r_upperarm',
#                 ('l_elbow', 'l_wrist'): 'l_forearm', ('r_elbow', 'r_wrist'): 'r_forearm',
#                 ('l_wrist', 'l_thumb'): 'l_thumb_len', ('r_wrist', 'r_thumb'): 'r_thumb_len',
#                 ('l_wrist', 'l_pinky'): 'l_pinky_len', ('r_wrist', 'r_pinky'): 'r_pinky_len',
#                 ('l_thumb', 'l_pinky'): 'l_hand_width', ('r_thumb', 'r_pinky'): 'r_hand_width',
#                 ('neck', 'l_shoulder'): 'l_collar', ('r_shoulder', 'neck'): 'r_collar',
#                 ('l_shoulder', 'l_hip'): 'l_lat', ('r_hip', 'r_shoulder'): 'r_lat',
#                 ('l_hip', 'midhip'): 'l_hipline', ('midhip', 'r_hip'): 'r_hipline',
#                 ('l_hip', 'l_knee'): 'l_thigh', ('r_hip', 'r_knee'): 'r_thigh',
#                 ('l_knee', 'l_ankle'): 'l_shin', ('r_knee', 'r_ankle'): 'r_shin',
#                 ('l_ankle', 'l_heel'): 'l_achilles', ('r_ankle', 'r_heel'): 'r_achilles',
#                 ('l_heel', 'l_bigtoe'): 'l_innerfoot', ('l_heel', 'l_bigtoe'): 'r_innerfoot',
#                 ('l_heel', 'l_smalltoe'): 'l_outerfoot', ('r_heel', 'r_smalltoe'): 'r_outerfoot',
#                 ('l_bigtoe', 'l_smalltoe'):'l_foot_width' , ('r_bigtoe', 'r_smalltoe'): 'r_foot_width'
#                 }
# # removed crown from limb names ('l_ear', 'crown'): 'l_scalp', ('r_ear', 'crown'): 'r_scalp',
#             #   ('nose', 'crown'): 'midhead'
duel_connections = list(limb_names.keys())

connections = [
    # Head to Neck
    ('nose', 'neck'),
    
    # Left Arm
    ('l_shoulder', 'l_elbow'),
    ('l_elbow', 'l_wrist'),
    ('l_wrist', 'l_thumb'),
    ('l_wrist', 'l_pinky'),
    ('l_thumb', 'l_pinky'),
    
    # Right Arm
    ('r_shoulder', 'r_elbow'),
    ('r_elbow', 'r_wrist'),
    ('r_wrist', 'r_thumb'),
    ('r_wrist', 'r_pinky'),
    ('r_thumb', 'r_pinky'),
    
    # Torso
    ('neck', 'l_shoulder'),
    ('l_shoulder', 'l_hip'),
    ('l_hip', 'midhip'),
    ('midhip', 'r_hip'),
    ('r_hip', 'r_shoulder'),
    ('r_shoulder', 'neck'),
    
    # Left Leg
    ('l_hip', 'l_knee'),
    ('l_knee', 'l_ankle'),
    ('l_ankle', 'l_heel'),
    ('l_heel', 'l_bigtoe'),
    ('l_heel', 'l_smalltoe'),
    ('l_bigtoe', 'l_smalltoe'),
    
    # Right Leg
    ('r_hip', 'r_knee'),
    ('r_knee', 'r_ankle'),
    ('r_ankle', 'r_heel'),
    ('r_heel', 'r_bigtoe'),
    ('r_heel', 'r_smalltoe'),
    ('r_bigtoe', 'r_smalltoe'),
]

def get_circles(radius, center, twoD = False):

    """
    Get the circle's points for the plot

    Parameters:
    - radius (float): radius of the circle
    - center (tuple): center of the circle
    - twoD (bool): if the plot is in 2D or 3D
    """
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    if twoD: 
        return x, y
    else :
        z = center[2] + np.zeros_like(theta)
        return x, y, z
    
def circle_equation(y, inv = True):

    """
    Get the equation of the three point line (for plot), hard-coded

    Parameters: 
    - y (float): y value
    - inv (bool): if we want the inside or the outside of the three point line
    """

    if inv:
        return ((-168)/69696)*(y**2) - 234
    else : 
        return ((168)/69696)*(y**2) + 234
#---------------------------------------------------------------------------------------
plt.rcParams['animation.embed_limit'] = 2**128

def plot_skeletons(df, width=1000, height=800, ball=True, aspect_mode='manual'):
    """
    Plots a 3D skeleton based on joint coordinates and connections.
    
    :param df: DataFrame containing the skeleton data.
    """
    fig = go.Figure()
    for index, row in df.iterrows():
        for conn in connections:
            # Constructing the full column names for the start and end joints
            start_x, start_y, start_z = f'{conn[0]}_x', f'{conn[0]}_y', f'{conn[0]}_z'
            end_x, end_y, end_z = f'{conn[1]}_x', f'{conn[1]}_y', f'{conn[1]}_z'
            
            # Adding the bone as a line between two joints
            fig.add_trace(
                go.Scatter3d(
                    x=[row[start_x], row[end_x]],
                    y=[row[start_y], row[end_y]],
                    z=[row[start_z], row[end_z]],
                    mode='lines+markers',
                    line=dict(width=3, color='blue'),
                    marker=dict(size=4, color='red'),
                    showlegend=False,
                )
            )
        if ball:
            fig.add_trace(
                    go.Scatter3d(
                        x=[row['ball_x']],
                        y=['ball_y'],
                        z=['ball_z'],
                        marker=dict(size=10, color='yellow'),
                        showlegend=False,
                    )
                )

    fig.update_layout(
        width=width, height=height,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectratio=dict(x=1,y=2,z=1),
        ),
        scene_aspectmode = aspect_mode,
        template = 'plotly_white',
        margin = dict(l=10,r=10,b=20,t=20)
    )

    fig.show()

#---------------------------------------------------------------------------------------

def gen_anim(df, connections=connections, el=25, az=90, size=6, filename=None, annotations=False, margin=0.5):
    """
    Create a 3D animation of a skeleton with dynamic axis limits to follow players.
    
    :param df: pandas DataFrame containing skeletal tracking data.
    :param connections: List of tuples defining the connections between joints.
    :param filename: Optional; if provided, the animation will be saved to this file.
    :param annotations: Boolean; if True, adds annotations to the plot.
    :param margin: Float; additional space around the players for the axis limits.
    """
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection='3d')
    #clear formatting
    ax.grid(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    
    player_colors = {player_id: plt.cm.tab10(i) for i, player_id in enumerate(df.player_id.unique())}
    lines = {player_id: [] for player_id in df.player_id.unique()}

    for player_id in df.player_id.unique():
        for connection in connections:
            line, = ax.plot([], [], [], 'o-', lw=2, markersize=5, color=player_colors[player_id])
            lines[player_id].append(line)
    if annotations:
        text = ax.text2D(0.05, 0.85, '', transform=ax.transAxes, fontsize=14)

    def init():
        for player_lines in lines.values():
            for line in player_lines:
                line.set_data([], [])
                line.set_3d_properties([])
        return [line for player_lines in lines.values() for line in player_lines]

    def animate(frame):
        frame_data = df[df['frame'] == frame]
        min_x, max_x = float('inf'), float('-inf')
        min_y, max_y = float('inf'), float('-inf')
        min_z, max_z = float('inf'), float('-inf')
        ax.view_init(elev=el, azim=az)
        if annotations:
            frame_text = []

        for player_id, player_lines in lines.items():
            player_data = frame_data[frame_data['player_id'] == player_id]
            if 'is_holding' in player_data.columns:
                current_color = 'red' if player_data['is_holding'].iloc[0] else player_colors[player_id]
            if not player_data.empty:
                if annotations and player_id == df.player_id.unique()[0]:
                    # frame_text.append(f"Frame: {frame} \nJoint Distance: {np.round(player_data['hold_joint_dist'].iloc[0],2)}m \nHolding? {player_data['is_holding'].iloc[0]}")
                    frame_text.append(f"Frame: {frame}")
                for line, (start, end) in zip(player_lines, connections):
                    xs = player_data[[f'{start}_x', f'{end}_x']].values.flatten()
                    ys = player_data[[f'{start}_y', f'{end}_y']].values.flatten()
                    zs = player_data[[f'{start}_z', f'{end}_z']].values.flatten()
                    line.set_data(xs, ys)
                    if 'is_holding' in player_data.columns:
                        line.set_color(current_color)
                    line.set_3d_properties(zs)
                    min_x, max_x = min(min_x, min(xs)), max(max_x, max(xs))
                    min_y, max_y = min(min_y, min(ys)), max(max_y, max(ys))
                    min_z, max_z = min(min_z, min(zs)), max(max_z, max(zs))
        
        # Update axis limits based on player positions plus some margin
        ax.set_xlim([min_x - margin, max_x + margin])
        ax.set_ylim([min_y - margin, max_y + margin])
        ax.set_zlim([min_z - margin, max_z + margin])

        if annotations:
            text.set_text("\n".join(frame_text))
        return [line for player_lines in lines.values() for line in player_lines] + ([text] if annotations else [])

    anim = FuncAnimation(fig, animate, init_func=init, frames=df.frame.unique(), interval=20, blit=True)

    plt.close(fig)
    out = HTML(anim.to_jshtml())
    if filename:
        with open(f'./visuals/{filename}.html', 'w') as f:
            f.write(out.data)

    return out

#---------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------
# plotly animation main function

def create_plotly_anim(df, ball_column='ball', joint_connections=limb_names, title=""):
    fig = go.Figure()

    # Function to add lines to the figure
    def add_lines(lines, links, fig, color='black'):
        for start, end in links:
            fig.add_trace(go.Scatter3d(
                x=[lines[start][0], lines[end][0]],
                y=[lines[start][1], lines[end][1]],
                z=[lines[start][2], lines[end][2]],
                mode='lines',
                line=dict(color=color),
                showlegend=False
            ))

    # Add court lines once
    out_lines = np.array([[-564, -300, 0], [-564, 300, 0], [564, 300, 0], [564, -300, 0]])
    links_lines = [(0, 1), (1, 2), (2, 3), (3, 0)]
    add_lines(out_lines, links_lines, fig)

    three_point_lines = np.array([[-564, -264, 0], [-402, -264, 0], [-564, 264, 0], [-402, 264, 0],
                                  [564, -264, 0], [402, -264, 0], [564, 264, 0], [402, 264, 0]])
    links_3plines = [(0, 1), (2, 3), (4, 5), (6, 7)]
    add_lines(three_point_lines, links_3plines, fig)

    box_lines = np.array([[-564, -72, 0], [-342, -72, 0], [-564, 72, 0], [-342, 72, 0],
                          [564, -72, 0], [342, -72, 0], [564, 72, 0], [342, 72, 0]])
    links_boxlines = [(0, 1), (2, 3), (1, 3), (4, 5), (6, 7), (5, 7)]
    add_lines(box_lines, links_boxlines, fig)

    basket_backboard = np.array([[-516, 31, 120], [-516, 31, 168], [-516, -31, 120], [-516, -31, 168],
                                 [516, 31, 120], [516, 31, 168], [516, -31, 120], [516, -31, 168]])
    links_backlines = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7)]
    add_lines(basket_backboard, links_backlines, fig)

    middle_line = np.array([[0, -300, 0], [0, 300, 0]])
    links_middle = [(0, 1)]
    add_lines(middle_line, links_middle, fig)

    circle3pts1 = np.array([[circle_equation(y), y, 0] for y in np.linspace(-264, 264, 529)])
    circle3pts2 = np.array([[circle_equation(y, inv=False), y, 0] for y in np.linspace(-264, 264, 529)])
    fig.add_trace(go.Scatter3d(
        x=circle3pts1[:, 0], y=circle3pts1[:, 1], z=circle3pts1[:, 2], 
        mode='markers', marker=dict(color='black', size=1),
        showlegend=False))
    fig.add_trace(go.Scatter3d(
        x=circle3pts2[:, 0], y=circle3pts2[:, 1], z=circle3pts2[:, 2],
          mode='markers', marker=dict(color='black', size=1),
          showlegend=False))

    x1, y1, z1 = get_circles(9, (-505, 0, 120))
    x2, y2, z2 = get_circles(9, (505, 0, 120))
    x3, y3, z3 = get_circles(72, (0, 0, 0))

    fig.add_trace(go.Scatter3d(x=x1, y=y1, z=z1, mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, mode='lines', line=dict(color='black'), showlegend=False))
    fig.add_trace(go.Scatter3d(x=x3, y=y3, z=z3, mode='lines', line=dict(color='black'), showlegend=False))

    player_ids = df.player_id.unique()

    # Get team ids and define color mapping
    team_ids = df.team_id.unique()
    color_mapping = {
        team_ids[0]: 'navy',
        team_ids[1]: 'green'
    }

    # Create a dictionary to map player_id to fullName
    player_name_mapping = {}
    for player_id in player_ids:
        player_name = df[df['player_id'] == player_id]['fullName'].iloc[0]
        player_name_mapping[player_id] = player_name


    # Initialize trace indices
    player_trace_indices = {}
    current_trace_index = len(fig.data)  # Start after the court lines

    # Add initial traces for each player
    for idx, player_id in enumerate(player_ids):
        team_id = df[df['player_id'] == player_id]['team_id'].iloc[0]
        player_color = color_mapping[team_id]
        player_name = player_name_mapping[player_id]
        trace = go.Scatter3d(
            x=[], y=[], z=[],
            mode='lines+markers',
            marker=dict(size=4, color=player_color),
            line=dict(width=2, color=player_color),
            name=f'Player: {player_name}'
        )
        fig.add_trace(trace)
        player_trace_indices[player_id] = current_trace_index
        current_trace_index += 1

    # Add initial trace for the ball
    if ball_column+"_x" in df.columns:
        trace = go.Scatter3d(
            x=[], y=[], z=[],
            mode='markers',
            marker=dict(size=6, color='red'),
            name='Ball'
        )
        fig.add_trace(trace)
        ball_trace_index = current_trace_index
        current_trace_index += 1
    else:
        ball_trace_index = None

    # Initialize the traces with data from the first frame
    initial_frame_data = df[df['frame'] == df['frame'].unique()[0]]

    for player_id in player_ids:
        player_data = initial_frame_data[initial_frame_data['player_id'] == player_id]
        if player_data.empty:
            x, y, z = [], [], []
        else:
            x, y, z = [], [], []
            for connection, (joint1, joint2) in joint_connections.items():
                x += [player_data[f'{joint1}_x'].values[0], player_data[f'{joint2}_x'].values[0], None]
                y += [player_data[f'{joint1}_y'].values[0], player_data[f'{joint2}_y'].values[0], None]
                z += [player_data[f'{joint1}_z'].values[0], player_data[f'{joint2}_z'].values[0], None]
        # Update the initial trace
        trace_index = player_trace_indices[player_id]
        fig.data[trace_index].update(x=x, y=y, z=z)

    if ball_column+"_x" in df.columns and ball_trace_index is not None:
        ball_data = initial_frame_data.iloc[0]
        x = [ball_data[f'{ball_column}_x']]
        y = [ball_data[f'{ball_column}_y']]
        z = [ball_data[f'{ball_column}_z']]
        fig.data[ball_trace_index].update(x=x, y=y, z=z)

    # Create frames to update the traces
    frames = []

    for frame_number, frame in enumerate(df['frame'].unique()):
        frame_data = df[df['frame'] == frame]
        frame_traces = []
        trace_indices = []

        # Update player traces
        for player_id in player_ids:
            player_data = frame_data[frame_data['player_id'] == player_id]
            if player_data.empty:
                x, y, z = [], [], []
            else:
                x, y, z = [], [], []
                for connection, (joint1, joint2) in joint_connections.items():
                    x += [player_data[f'{joint1}_x'].values[0], player_data[f'{joint2}_x'].values[0], None]
                    y += [player_data[f'{joint1}_y'].values[0], player_data[f'{joint2}_y'].values[0], None]
                    z += [player_data[f'{joint1}_z'].values[0], player_data[f'{joint2}_z'].values[0], None]
            # Create a data dictionary and specify the type
            trace = dict(type='scatter3d', x=x, y=y, z=z)
            frame_traces.append(trace)
            trace_indices.append(player_trace_indices[player_id])

        # Update ball trace
        if ball_column+"_x" in df.columns and ball_trace_index is not None:
            ball_frame_data = frame_data.iloc[0]
            x = [ball_frame_data[f'{ball_column}_x']]
            y = [ball_frame_data[f'{ball_column}_y']]
            z = [ball_frame_data[f'{ball_column}_z']]
            trace = dict(type='scatter3d', x=x, y=y, z=z)
            frame_traces.append(trace)
            trace_indices.append(ball_trace_index)
        
        # Get TimeUTC for the current frame
        time_utc = frame_data['timeUTC'].iloc[0]

        # Create the frame
        frames.append(go.Frame(
            data=frame_traces,
            traces=trace_indices,
            name=f'frame{frame_number}',
            layout=go.Layout(
                annotations=[
                    dict(
                        text=f"Frame: {frame}",
                        x=0.05,
                        y=0.95,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=16, color="black")
                    ),
                    dict( 
                        text=f"TimeUTC: {time_utc}",
                        x=0.05,
                        y=0.90,
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=16, color="black")
                    )
                ]
            )
        ))

    fig.frames = frames

    # Update layout with animation controls
    fig.update_layout(
        height=800,
        width=1000,
        template='plotly_white',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='manual',
            aspectratio=dict(x=3, y=2, z=1)
        ),
        title=title,
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 100, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300}
                        }]
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0}
                            }
                        ]
                    ),
                    dict(
                        label="Next Frame",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0}
                            }
                        ]
                    ),
                ],
                showactive=False,
                x=0.1,
                y=0,
                xanchor="right",
                yanchor="top"
            )
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [f.name],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate"
                            }
                        ],
                        "label": f'Frame {i}',
                        "method": "animate"
                    }
                    for i, f in enumerate(fig.frames)
                ],
                "currentvalue": {"prefix": "Frame: "},
                "pad": {"b": 10, "t": 50},
                "x": 0.1,
                "len": 0.9
            }
        ]
    )

    return fig