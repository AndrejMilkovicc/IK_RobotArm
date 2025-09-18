import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, TextBox, Slider, CheckButtons
import time
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D

class RoboticArm:
    def __init__(self, segment_lengths, angle_limits=None, dimension=2):
        """
        Initialize the robotic arm with given segment lengths and angle limits.
        
        Parameters:
        segment_lengths (list): List of lengths for each arm segment
        angle_limits (list): List of (min, max) angle limits for each joint
        dimension (int): 2 for 2D, 3 for 3D
        """
        self.segment_lengths = np.array(segment_lengths)
        self.num_segments = len(segment_lengths)
        self.dimension = dimension
        
        # Set angle limits (default to ±π for all joints)
        if angle_limits is None:
            if dimension == 2:
                self.angle_limits = [(-np.pi, np.pi)] * self.num_segments
            else:
                # For 3D, we need limits for each of the 3 rotational axes per joint
                self.angle_limits = [(-np.pi, np.pi)] * (self.num_segments * 3)
        else:
            self.angle_limits = angle_limits
            
        # Initialize angles based on dimension
        if dimension == 2:
            self.angles = np.zeros(self.num_segments)
        else:
            # For 3D, each joint has 3 rotational angles (roll, pitch, yaw)
            self.angles = np.zeros(self.num_segments * 3)
            
        self.joint_positions = None
        self.history = []  # Store history of positions for animation
        self.max_history_frames = 200  # Limit history size to prevent memory issues
        self.workspace_resolution = 50  # Default resolution for workspace calculation
        self.collision_plane = None  # Collision plane (e.g., y=0 for table)
        self.update_joint_positions()
        
    def update_joint_positions(self):
        """Calculate the positions of all joints based on current angles (optimized)."""
        if self.dimension == 2:
            self.joint_positions = np.zeros((self.num_segments + 1, 2))
            
            # Precompute cumulative angles for efficiency
            cumulative_angles = np.cumsum(self.angles)
            
            # Calculate x and y positions using vectorized operations
            x_positions = np.cumsum(self.segment_lengths * np.cos(cumulative_angles))
            y_positions = np.cumsum(self.segment_lengths * np.sin(cumulative_angles))
            
            # Set joint positions
            self.joint_positions[1:, 0] = x_positions
            self.joint_positions[1:, 1] = y_positions
        else:
            # 3D case - each joint has 3 rotational degrees of freedom
            self.joint_positions = np.zeros((self.num_segments + 1, 3))
            
            # Start with identity transformation
            transformation = np.eye(4)
            
            for i in range(self.num_segments):
                # Get rotation angles for this joint
                roll, pitch, yaw = self.angles[i*3:(i+1)*3]
                
                # Create rotation matrices
                Rx = np.array([[1, 0, 0, 0],
                              [0, np.cos(roll), -np.sin(roll), 0],
                              [0, np.sin(roll), np.cos(roll), 0],
                              [0, 0, 0, 1]])
                
                Ry = np.array([[np.cos(pitch), 0, np.sin(pitch), 0],
                              [0, 1, 0, 0],
                              [-np.sin(pitch), 0, np.cos(pitch), 0],
                              [0, 0, 0, 1]])
                
                Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                              [np.sin(yaw), np.cos(yaw), 0, 0],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
                
                # Combine rotations
                rotation = Rz @ Ry @ Rx
                
                # Translation for this segment
                translation = np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, self.segment_lengths[i]],
                                       [0, 0, 0, 1]])
                
                # Update transformation
                transformation = transformation @ rotation @ translation
                
                # Extract position
                self.joint_positions[i+1] = transformation[:3, 3]
        
        # Check for collisions
        if self.collision_plane is not None and self.check_collision():
            # Move joints above collision plane
            for i in range(1, self.num_segments + 1):
                if self.joint_positions[i, 1] < self.collision_plane:
                    self.joint_positions[i, 1] = self.collision_plane + 0.01
        
        # Save current state to history (with size limit)
        if len(self.history) >= self.max_history_frames:
            self.history.pop(0)
        self.history.append((self.angles.copy(), self.joint_positions.copy()))
    
    def check_collision(self):
        """Check if any joint is below the collision plane."""
        if self.collision_plane is None:
            return False
            
        for i in range(1, self.num_segments + 1):
            if self.joint_positions[i, 1] < self.collision_plane:
                return True
                
        return False
    
    def forward_kinematics(self):
        """Compute the end effector position using forward kinematics."""
        return self.joint_positions[-1]
    
    def is_reachable(self, target):
        """Check if a target is reachable by the arm."""
        max_reach = np.sum(self.segment_lengths)
        min_reach = max(0, np.max(self.segment_lengths) - np.sum(np.delete(self.segment_lengths, 
                                                                          np.argmax(self.segment_lengths))))
        distance = np.linalg.norm(target)
        return min_reach <= distance <= max_reach
    
    def jacobian(self):
        """Compute the Jacobian matrix for the current arm configuration."""
        if self.dimension == 2:
            jac = np.zeros((2, self.num_segments))
            end_effector = self.forward_kinematics()
            
            for i in range(self.num_segments):
                # Jacobian for a revolute joint
                joint_pos = self.joint_positions[i]
                jac[0, i] = -(end_effector[1] - joint_pos[1])  # -dy/dθ
                jac[1, i] = end_effector[0] - joint_pos[0]     # dx/dθ
        else:
            # 3D Jacobian - more complex
            jac = np.zeros((6, self.num_segments * 3))  # 6DOF (3 position + 3 orientation)
            end_effector = self.forward_kinematics()
            
            for i in range(self.num_segments):
                joint_pos = self.joint_positions[i]
                
                # Position Jacobian
                jac[0:3, i*3:(i+1)*3] = np.eye(3)  # Linear velocity components
                
                # Orientation Jacobian (simplified)
                jac[3:6, i*3:(i+1)*3] = np.eye(3)  # Angular velocity components
                
        return jac
    
    def apply_angle_constraints(self):
        """Apply angle constraints to keep joints within limits."""
        for i in range(len(self.angles)):
            min_angle, max_angle = self.angle_limits[i]
            self.angles[i] = np.clip(self.angles[i], min_angle, max_angle)
    
    def inverse_kinematics(self, target, learning_rate=0.1, damping=0.01, 
                          threshold=1e-3, max_iter=1000):
        """
        Use damped Jacobian transpose method to solve inverse kinematics.
        
        Parameters:
        target (array): Target position [x, y] or [x, y, z]
        learning_rate (float): Step size for gradient descent
        damping (float): Damping factor for numerical stability
        threshold (float): Convergence threshold
        max_iter (int): Maximum number of iterations
        
        Returns:
        bool: True if converged, False otherwise
        """
        # Check if target is reachable
        if not self.is_reachable(target):
            return False
        
        self.history = []  # Reset history
        self.history.append((self.angles.copy(), self.joint_positions.copy()))
        
        for _ in range(max_iter):
            # Current end effector position
            current_pos = self.forward_kinematics()
            
            # Error vector
            error = target - current_pos
            
            # Check if we're close enough
            if np.linalg.norm(error) < threshold:
                return True
                
            # Compute Jacobian
            J = self.jacobian()
            
            # Damped Jacobian transpose method for stability
            if self.dimension == 2:
                JJT = J @ J.T
                damping_matrix = damping * damping * np.eye(2)
                inv_matrix = np.linalg.inv(JJT + damping_matrix)
                
                # Update angles using damped Jacobian transpose method
                delta_theta = learning_rate * J.T @ inv_matrix @ error
            else:
                # For 3D, we need to handle the 6DOF case
                JJT = J @ J.T
                damping_matrix = damping * damping * np.eye(6)
                inv_matrix = np.linalg.inv(JJT + damping_matrix)
                
                # Extend error to 6DOF (we only care about position for now)
                error_6dof = np.zeros(6)
                error_6dof[:3] = error
                delta_theta = learning_rate * J.T @ inv_matrix @ error_6dof
            
            # Apply the angle changes
            self.angles += delta_theta
            
            # Apply angle constraints
            self.apply_angle_constraints()
            
            # Update joint positions
            self.update_joint_positions()
            
        return False
    
    def follow_path(self, path_points, learning_rate=0.1, damping=0.01, stream=False):
        """
        Make the arm follow a path defined by a series of points.
        
        Parameters:
        path_points (list): List of target points to follow
        learning_rate (float): Step size for gradient descent
        damping (float): Damping factor for numerical stability
        stream (bool): If True, stream points one by one with animation
        """
        if stream:
            # Stream points one by one with animation
            for point in path_points:
                success = self.inverse_kinematics(point, learning_rate, damping)
                if not success:
                    print(f"Failed to reach point {point}")
                # Add a small delay to make the animation visible
                time.sleep(0.1)
        else:
            # Process all points first, then animate
            for point in path_points:
                success = self.inverse_kinematics(point, learning_rate, damping)
                if not success:
                    print(f"Failed to reach point {point}")
    
    def get_segments(self):
        """Get line segments for visualization."""
        segments = []
        for i in range(self.num_segments):
            segments.append([self.joint_positions[i], self.joint_positions[i+1]])
        return segments
    
    def calculate_workspace(self, resolution=None):
        """
        Calculate the reachable workspace of the robotic arm.
        
        Parameters:
        resolution (int): Resolution of the workspace grid
        
        Returns:
        tuple: (x_grid, y_grid, reachable_mask) representing the workspace
        """
        if resolution is None:
            resolution = self.workspace_resolution
            
        max_reach = np.sum(self.segment_lengths)
        
        if self.dimension == 2:
            x = np.linspace(-max_reach, max_reach, resolution)
            y = np.linspace(-max_reach, max_reach, resolution)
            x_grid, y_grid = np.meshgrid(x, y)
            
            # Create a mask of reachable points
            distances = np.sqrt(x_grid**2 + y_grid**2)
            min_reach = max(0, np.max(self.segment_lengths) - 
                           np.sum(np.delete(self.segment_lengths, 
                                           np.argmax(self.segment_lengths))))
            reachable_mask = (distances >= min_reach) & (distances <= max_reach)
            
            return x_grid, y_grid, reachable_mask
        else:
            # 3D workspace - simplified to 2D projection for visualization
            x = np.linspace(-max_reach, max_reach, resolution)
            z = np.linspace(-max_reach, max_reach, resolution)
            x_grid, z_grid = np.meshgrid(x, z)
            
            # Create a mask of reachable points (projection on xz-plane)
            distances = np.sqrt(x_grid**2 + z_grid**2)
            min_reach = max(0, np.max(self.segment_lengths) - 
                           np.sum(np.delete(self.segment_lengths, 
                                           np.argmax(self.segment_lengths))))
            reachable_mask = (distances >= min_reach) & (distances <= max_reach)
            
            return x_grid, z_grid, reachable_mask

class ArmVisualizer:
    def __init__(self, arm):
        """
        Initialize the visualizer for the robotic arm.
        
        Parameters:
        arm: RoboticArm instance
        """
        self.arm = arm
        self.fig = plt.figure(figsize=(12, 10))
        
        if arm.dimension == 2:
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax = self.fig.add_subplot(111, projection='3d')
            
        self.target = np.array([1.5, 1.5, 0] if arm.dimension == 3 else [1.5, 1.5])
        self.animation = None
        self.show_workspace = False
        self.workspace_plot = None
        self.workspace_resolution = 50
        
        # Set up the plot
        self.setup_plot()
        
        # Add UI controls
        self.setup_ui()
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def setup_plot(self):
        """Set up the initial plot configuration."""
        max_length = np.sum(self.arm.segment_lengths)
        
        if self.arm.dimension == 2:
            self.ax.set_xlim(-max_length * 1.2, max_length * 1.2)
            self.ax.set_ylim(-max_length * 1.2, max_length * 1.2)
            self.ax.set_aspect('equal')
            self.ax.grid(True)
            self.ax.set_title('Robotic Arm Simulator (2D)')
            self.ax.set_xlabel('X position')
            self.ax.set_ylabel('Y position')
            
            # Draw collision plane if exists
            if self.arm.collision_plane is not None:
                self.ax.axhline(y=self.arm.collision_plane, color='r', linestyle='--', alpha=0.5, label='Collision Plane')
        else:
            self.ax.set_xlim(-max_length * 1.2, max_length * 1.2)
            self.ax.set_ylim(-max_length * 1.2, max_length * 1.2)
            self.ax.set_zlim(-max_length * 1.2, max_length * 1.2)
            self.ax.grid(True)
            self.ax.set_title('Robotic Arm Simulator (3D)')
            self.ax.set_xlabel('X position')
            self.ax.set_ylabel('Y position')
            self.ax.set_zlabel('Z position')
            
            # Draw collision plane if exists
            if self.arm.collision_plane is not None:
                # Create a mesh grid for the collision plane
                xx, zz = np.meshgrid(np.linspace(-max_length, max_length, 2), 
                                    np.linspace(-max_length, max_length, 2))
                yy = np.ones_like(xx) * self.arm.collision_plane
                self.ax.plot_surface(xx, yy, zz, alpha=0.2, color='r')
        
        # Draw target point
        if self.arm.dimension == 2:
            self.target_point = self.ax.plot(self.target[0], self.target[1], 'ro', 
                                             markersize=10, label='Target')[0]
        else:
            self.target_point = self.ax.plot([self.target[0]], [self.target[1]], [self.target[2]], 
                                             'ro', markersize=10, label='Target')[0]
        
        # Draw initial arm position
        segments = self.arm.get_segments()
        self.lines = []
        
        if self.arm.dimension == 2:
            for i, segment in enumerate(segments):
                line, = self.ax.plot([segment[0][0], segment[1][0]], 
                                    [segment[0][1], segment[1][1]], 
                                    'o-', linewidth=3, markersize=8, 
                                    label=f'Segment {i+1}' if i == 0 else "")
                self.lines.append(line)
            
            # Draw joints
            joints = self.arm.joint_positions
            self.joint_plot = self.ax.plot(joints[:, 0], joints[:, 1], 'o', 
                                           markersize=6, color='black')[0]
        else:
            for i, segment in enumerate(segments):
                line = Line3D([segment[0][0], segment[1][0]], 
                             [segment[0][1], segment[1][1]], 
                             [segment[0][2], segment[1][2]], 
                             marker='o', markersize=6, linewidth=3)
                self.ax.add_line(line)
                self.lines.append(line)
            
            # Draw joints
            joints = self.arm.joint_positions
            self.joint_plot = self.ax.plot(joints[:, 0], joints[:, 1], joints[:, 2], 
                                           'o', markersize=6, color='black')[0]
        
        # Add legend
        self.ax.legend(loc='upper left')
        
    def setup_ui(self):
        """Set up user interface controls."""
        # Create text boxes for target input
        y_pos = 0.05 if self.arm.dimension == 2 else 0.1
        ax_x = plt.axes([0.15, y_pos, 0.1, 0.04])
        ax_y = plt.axes([0.35, y_pos, 0.1, 0.04])
        
        if self.arm.dimension == 3:
            ax_z = plt.axes([0.55, y_pos, 0.1, 0.04])
        
        self.x_textbox = TextBox(ax_x, 'X', initial=str(self.target[0]))
        self.y_textbox = TextBox(ax_y, 'Y', initial=str(self.target[1]))
        
        if self.arm.dimension == 3:
            self.z_textbox = TextBox(ax_z, 'Z', initial=str(self.target[2]))
        
        # Create buttons
        ax_go = plt.axes([0.15, y_pos - 0.05, 0.1, 0.04])
        ax_reset = plt.axes([0.35, y_pos - 0.05, 0.1, 0.04])
        ax_workspace = plt.axes([0.55, y_pos - 0.05, 0.1, 0.04])
        ax_circle = plt.axes([0.75, y_pos - 0.05, 0.1, 0.04])
        ax_square = plt.axes([0.15, y_pos - 0.1, 0.1, 0.04])
        ax_stream = plt.axes([0.35, y_pos - 0.1, 0.1, 0.04])
        ax_3d_toggle = plt.axes([0.55, y_pos - 0.1, 0.1, 0.04])
        
        self.go_button = Button(ax_go, 'Go to Target')
        self.reset_button = Button(ax_reset, 'Reset Arm')
        self.workspace_button = Button(ax_workspace, 'Toggle Workspace')
        self.circle_button = Button(ax_circle, 'Follow Circle')
        self.square_button = Button(ax_square, 'Follow Square')
        self.stream_button = Button(ax_stream, 'Stream Path')
        self.toggle_3d_button = Button(ax_3d_toggle, '2D/3D Toggle')
        
        # Create damping slider
        ax_damping = plt.axes([0.75, y_pos - 0.1, 0.2, 0.04])
        self.damping_slider = Slider(ax_damping, 'Damping', 0.001, 0.1, valinit=0.01)
        
        # Create resolution slider
        ax_resolution = plt.axes([0.75, y_pos - 0.15, 0.2, 0.04])
        self.resolution_slider = Slider(ax_resolution, 'Resolution', 10, 100, valinit=50, valfmt='%0.0f')
        
        # Connect events
        self.go_button.on_clicked(self.on_go_clicked)
        self.reset_button.on_clicked(self.on_reset_clicked)
        self.workspace_button.on_clicked(self.on_workspace_clicked)
        self.circle_button.on_clicked(self.on_circle_clicked)
        self.square_button.on_clicked(self.on_square_clicked)
        self.stream_button.on_clicked(self.on_stream_clicked)
        self.toggle_3d_button.on_clicked(self.on_toggle_3d_clicked)
        self.resolution_slider.on_changed(self.on_resolution_changed)
        
    def on_go_clicked(self, event):
        """Handle Go button click event."""
        try:
            x = float(self.x_textbox.text)
            y = float(self.y_textbox.text)
            
            if self.arm.dimension == 2:
                self.target = np.array([x, y])
                self.target_point.set_data([x], [y])
            else:
                z = float(self.z_textbox.text)
                self.target = np.array([x, y, z])
                self.target_point.set_data([x], [y])
                self.target_point.set_3d_properties([z])
            
            # Reset arm and run IK
            if self.arm.dimension == 2:
                self.arm.angles = np.zeros(self.arm.num_segments)
            else:
                self.arm.angles = np.zeros(self.arm.num_segments * 3)
                
            self.arm.update_joint_positions()
            
            damping = self.damping_slider.val
            success = self.arm.inverse_kinematics(self.target, damping=damping)
            
            if not success:
                print("Target is unreachable or algorithm did not converge")
            
            # Animate the movement
            self.animate()
            
        except ValueError:
            print("Please enter valid numbers for coordinates")
    
    def on_reset_clicked(self, event):
        """Handle Reset button click event."""
        if self.arm.dimension == 2:
            self.arm.angles = np.zeros(self.arm.num_segments)
        else:
            self.arm.angles = np.zeros(self.arm.num_segments * 3)
            
        self.arm.update_joint_positions()
        self.update_plot()
    
    def on_workspace_clicked(self, event):
        """Toggle workspace visualization."""
        self.show_workspace = not self.show_workspace
        
        if self.show_workspace:
            # Calculate and show workspace
            resolution = int(self.resolution_slider.val)
            if self.arm.dimension == 2:
                x_grid, y_grid, reachable_mask = self.arm.calculate_workspace(resolution)
                self.workspace_plot = self.ax.contourf(x_grid, y_grid, reachable_mask, 
                                                      levels=1, alpha=0.2, cmap='cool')
            else:
                x_grid, z_grid, reachable_mask = self.arm.calculate_workspace(resolution)
                # For 3D, we need to handle visualization differently
                self.workspace_plot = self.ax.contourf(x_grid, reachable_mask, z_grid, 
                                                      levels=1, alpha=0.2, cmap='cool')
        else:
            # Remove workspace visualization
            if self.workspace_plot:
                if self.arm.dimension == 2:
                    for coll in self.workspace_plot.collections:
                        coll.remove()
                else:
                    # Handle 3D case
                    pass
                self.workspace_plot = None
        
        self.fig.canvas.draw()
    
    def on_resolution_changed(self, val):
        """Handle resolution slider change."""
        self.arm.workspace_resolution = int(val)
        
        # Update workspace if it's currently shown
        if self.show_workspace:
            self.on_workspace_clicked(None)
            self.on_workspace_clicked(None)
    
    def generate_path(self, shape):
        """Generate a path based on the specified shape."""
        if shape == 'circle':
            center = np.array([1.0, 0.0, 0.0] if self.arm.dimension == 3 else [1.0, 0.0])
            radius = 1.0
            num_points = 20
            
            # Generate circular path
            angles = np.linspace(0, 2*np.pi, num_points)
            if self.arm.dimension == 2:
                path_points = [center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles]
            else:
                path_points = [center + radius * np.array([np.cos(a), 0, np.sin(a)]) for a in angles]
                
        elif shape == 'square':
            center = np.array([1.0, 0.0, 0.0] if self.arm.dimension == 3 else [1.0, 0.0])
            side_length = 1.0
            num_points_per_side = 5
            
            # Generate square path
            path_points = []
            
            if self.arm.dimension == 2:
                # Bottom side
                for i in range(num_points_per_side):
                    x = center[0] - side_length/2 + (side_length * i / (num_points_per_side - 1))
                    y = center[1] - side_length/2
                    path_points.append(np.array([x, y]))
                
                # Right side
                for i in range(1, num_points_per_side):
                    x = center[0] + side_length/2
                    y = center[1] - side_length/2 + (side_length * i / (num_points_per_side - 1))
                    path_points.append(np.array([x, y]))
                
                # Top side
                for i in range(1, num_points_per_side):
                    x = center[0] + side_length/2 - (side_length * i / (num_points_per_side - 1))
                    y = center[1] + side_length/2
                    path_points.append(np.array([x, y]))
                
                # Left side
                for i in range(1, num_points_per_side - 1):
                    x = center[0] - side_length/2
                    y = center[1] + side_length/2 - (side_length * i / (num_points_per_side - 1))
                    path_points.append(np.array([x, y]))
            else:
                # 3D square path (in XZ plane)
                # Bottom side
                for i in range(num_points_per_side):
                    x = center[0] - side_length/2 + (side_length * i / (num_points_per_side - 1))
                    z = center[2] - side_length/2
                    path_points.append(np.array([x, center[1], z]))
                
                # Right side
                for i in range(1, num_points_per_side):
                    x = center[0] + side_length/2
                    z = center[2] - side_length/2 + (side_length * i / (num_points_per_side - 1))
                    path_points.append(np.array([x, center[1], z]))
                
                # Top side
                for i in range(1, num_points_per_side):
                    x = center[0] + side_length/2 - (side_length * i / (num_points_per_side - 1))
                    z = center[2] + side_length/2
                    path_points.append(np.array([x, center[1], z]))
                
                # Left side
                for i in range(1, num_points_per_side - 1):
                    x = center[0] - side_length/2
                    z = center[2] + side_length/2 - (side_length * i / (num_points_per_side - 1))
                    path_points.append(np.array([x, center[1], z]))
                
        return path_points
    
    def on_circle_clicked(self, event):
        """Make the arm follow a circular path."""
        path_points = self.generate_path('circle')
        
        # Reset arm
        if self.arm.dimension == 2:
            self.arm.angles = np.zeros(self.arm.num_segments)
        else:
            self.arm.angles = np.zeros(self.arm.num_segments * 3)
            
        self.arm.update_joint_positions()
        
        # Follow path
        damping = self.damping_slider.val
        self.arm.follow_path(path_points, damping=damping, stream=False)
        
        # Animate the movement
        self.animate()
    
    def on_square_clicked(self, event):
        """Make the arm follow a square path."""
        path_points = self.generate_path('square')
        
        # Reset arm
        if self.arm.dimension == 2:
            self.arm.angles = np.zeros(self.arm.num_segments)
        else:
            self.arm.angles = np.zeros(self.arm.num_segments * 3)
            
        self.arm.update_joint_positions()
        
        # Follow path
        damping = self.damping_slider.val
        self.arm.follow_path(path_points, damping=damping, stream=False)
        
        # Animate the movement
        self.animate()
    
    def on_stream_clicked(self, event):
        """Make the arm follow a circular path with streaming."""
        path_points = self.generate_path('circle')
        
        # Reset arm
        if self.arm.dimension == 2:
            self.arm.angles = np.zeros(self.arm.num_segments)
        else:
            self.arm.angles = np.zeros(self.arm.num_segments * 3)
            
        self.arm.update_joint_positions()
        
        # Follow path with streaming
        damping = self.damping_slider.val
        self.arm.follow_path(path_points, damping=damping, stream=True)
        
        # Update plot after streaming
        self.update_plot()
    
    def on_toggle_3d_clicked(self, event):
        """Toggle between 2D and 3D mode."""
        # This would require recreating the arm and visualizer
        print("2D/3D toggle would require restarting the application")
    
    def on_key_press(self, event):
        """Handle keyboard events for moving the target."""
        step = 0.1
        
        if event.key == 'up':
            self.target[1] += step
        elif event.key == 'down':
            self.target[1] -= step
        elif event.key == 'left':
            self.target[0] -= step
        elif event.key == 'right':
            self.target[0] += step
        elif event.key == 'pageup' and self.arm.dimension == 3:
            self.target[2] += step
        elif event.key == 'pagedown' and self.arm.dimension == 3:
            self.target[2] -= step
        else:
            return
        
        # Update text boxes
        self.x_textbox.set_val(str(round(self.target[0], 2)))
        self.y_textbox.set_val(str(round(self.target[1], 2)))
        
        if self.arm.dimension == 3:
            self.z_textbox.set_val(str(round(self.target[2], 2)))
        
        # Update target point
        if self.arm.dimension == 2:
            self.target_point.set_data([self.target[0]], [self.target[1]])
        else:
            self.target_point.set_data([self.target[0]], [self.target[1]])
            self.target_point.set_3d_properties([self.target[2]])
        
        # Move arm to new target
        damping = self.damping_slider.val
        success = self.arm.inverse_kinematics(self.target, damping=damping)
        
        if not success:
            print("Target is unreachable or algorithm did not converge")
        
        # Animate the movement
        self.animate()
        
        self.fig.canvas.draw()
    
    def update_plot(self):
        """Update the plot with current arm configuration."""
        segments = self.arm.get_segments()
        
        if self.arm.dimension == 2:
            for i, segment in enumerate(segments):
                self.lines[i].set_data([segment[0][0], segment[1][0]], 
                                      [segment[0][1], segment[1][1]])
            
            self.joint_plot.set_data(self.arm.joint_positions[:, 0], 
                                    self.arm.joint_positions[:, 1])
        else:
            for i, segment in enumerate(segments):
                self.lines[i].set_data([segment[0][0], segment[1][0]], 
                                      [segment[0][1], segment[1][1]])
                self.lines[i].set_3d_properties([segment[0][2], segment[1][2]])
            
            self.joint_plot.set_data(self.arm.joint_positions[:, 0], 
                                    self.arm.joint_positions[:, 1])
            self.joint_plot.set_3d_properties(self.arm.joint_positions[:, 2])
            
        self.fig.canvas.draw()
    
    def animate(self):
        """Animate the arm movement."""
        if self.animation:
            self.animation.event_source.stop()
        
        frames = len(self.arm.history)
        
        def update(frame):
            angles, joint_positions = self.arm.history[frame]
            self.arm.angles = angles
            self.arm.joint_positions = joint_positions
            
            segments = self.arm.get_segments()
            
            if self.arm.dimension == 2:
                for i, segment in enumerate(segments):
                    self.lines[i].set_data([segment[0][0], segment[1][0]], 
                                          [segment[0][1], segment[1][1]])
                
                self.joint_plot.set_data(joint_positions[:, 0], joint_positions[:, 1])
            else:
                for i, segment in enumerate(segments):
                    self.lines[i].set_data([segment[0][0], segment[1][0]], 
                                          [segment[0][1], segment[1][1]])
                    self.lines[i].set_3d_properties([segment[0][2], segment[1][2]])
                
                self.joint_plot.set_data(joint_positions[:, 0], joint_positions[:, 1])
                self.joint_plot.set_3d_properties(joint_positions[:, 2])
            
            return self.lines + [self.joint_plot, self.target_point]
        
        self.animation = FuncAnimation(self.fig, update, frames=frames, 
                                      interval=50, blit=True, repeat=False)
        self.fig.canvas.draw()

def main():
    """Run the robotic arm simulation."""
    # Create a robotic arm with 4 segments of different lengths and angle limits
    segment_lengths = [1.0, 0.8, 0.6, 0.4]
    angle_limits = [(-np.pi/2, np.pi/2),  # Base joint limited to ±90°
                    (-np.pi/2, np.pi/2),  # Second joint limited to ±90°
                    (-np.pi/2, np.pi/2),  # Third joint limited to ±90°
                    (-np.pi/2, np.pi/2)]  # Fourth joint limited to ±90°
    
    # Choose 2D or 3D mode
    dimension = 2  # Change to 3 for 3D mode
    
    arm = RoboticArm(segment_lengths, angle_limits, dimension)
    
    # Set collision plane (e.g., y=0 for table)
    arm.collision_plane = 0.0
    
    # Create and show the visualizer
    visualizer = ArmVisualizer(arm)
    plt.show()

if __name__ == "__main__":
    main()
    
    # Create and show the visualizer
    visualizer = ArmVisualizer(arm)
    plt.show()

if __name__ == "__main__":
    main()