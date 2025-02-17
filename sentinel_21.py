import rps.robotarium as robotarium
import numpy as np
import matplotlib.pyplot as plt
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *

class FireZone:
    def __init__(self, center, radius=0.3):
        self.center = np.array(center)
        self.radius = radius
    
    def is_in_zone(self, position):
        return np.linalg.norm(position - self.center) <= self.radius

class ThermalCamera:
    def __init__(self, resolution=(100, 100), fov=90, view_distance=1.5):
        self.resolution = resolution
        self.fov = fov
        self.view_distance = view_distance
        
    def capture(self, position, orientation, fire_center, fire_radius):
        """Generate thermal image using NumPy operations"""
        thermal_image = np.zeros(self.resolution)
        rel_pos = fire_center - position
        distance = np.linalg.norm(rel_pos)
        
        if distance < self.view_distance:
            rel_angle = np.arctan2(rel_pos[1], rel_pos[0]) - orientation
            rel_angle = np.rad2deg(rel_angle)
            
            if abs(rel_angle) < self.fov/2:
                x_img = int((rel_angle + self.fov/2) / self.fov * self.resolution[0])
                y_img = int((1 - distance/self.view_distance) * self.resolution[1])
                
                y_grid, x_grid = np.mgrid[0:self.resolution[1], 0:self.resolution[0]]
                dist_from_center = np.sqrt((x_grid - x_img)**2 + (y_grid - y_img)**2)
                radius = int(50 * (1 - 0.5*distance/self.view_distance))
                
                circle_mask = dist_from_center <= radius
                thermal_image[circle_mask] = 1.0
                
                glow_radius = radius * 2
                glow_mask = (dist_from_center <= glow_radius) & (~circle_mask)
                if np.any(glow_mask):
                    glow_intensity = 0.5 * (1 - dist_from_center[glow_mask]/glow_radius)
                    thermal_image[glow_mask] = glow_intensity
        
        return thermal_image

class Robot:
    def __init__(self, id, is_leader=False):
        self.id = id
        self.is_leader = is_leader
        self.detected_fire = False
        self.circling = False
        self.circle_radius = 0.5
        self.current_angle = 0
        self.last_position = None
        self.stall_count = 0
        self.camera = ThermalCamera() if is_leader else None
        self.thermal_image = None
        self.in_formation = False
        self.reached_leader = False
        self.lock_position = None
        self.optimal_distance = 0.4
        self.approach_speed = 0.2 if is_leader else 0.15  # Faster leader
        self.cruise_speed = 0.17 if is_leader else 0.12   # Faster cruising
        
    def update_position(self, position):
        if self.last_position is not None:
            movement = np.linalg.norm(position - self.last_position)
            if movement < 0.01:
                self.stall_count += 1
            else:
                self.stall_count = 0
        self.last_position = position.copy()
    
    def is_stalled(self):
        return self.stall_count > 20
        
    def compute_optimal_position(self, fire_center, current_pos):
        """Compute optimal observation position for leader"""
        if self.lock_position is None:
            self.lock_position = current_pos.copy()
            
        to_fire = fire_center - current_pos
        distance = np.linalg.norm(to_fire)
        
        if distance > 0:
            optimal_pos = fire_center - (to_fire / distance) * self.optimal_distance
            return optimal_pos
        return current_pos
        
    def compute_circling_position(self, fire_center, num_robots):
        if self.is_leader and self.detected_fire:
            # Tighter circling for leader
            radius_variation = 0.05 * np.sin(self.current_angle)
            actual_radius = self.optimal_distance + radius_variation
            angle = self.current_angle
        else:
            # Normal circling for followers
            radius_variation = 0.1 * np.sin(self.current_angle)
            actual_radius = self.circle_radius + radius_variation
            angle = self.current_angle
        
        x = fire_center[0] + actual_radius * np.cos(angle)
        y = fire_center[1] + actual_radius * np.sin(angle)
        return np.array([x, y])

    def check_reached_leader(self, position, leader_position, threshold=0.3):
        """Check if follower has reached leader's vicinity"""
        if not self.is_leader and not self.reached_leader:
            distance = np.linalg.norm(position - leader_position)
            if distance < threshold:
                self.reached_leader = True
                return True
        return False

def compute_targets(poses, iteration_count, robots, fire_zone, leader_index):
    """Compute target positions with improved leader tracking"""
    targets = np.zeros((2, len(robots)))
    leader_pos = poses[:2, leader_index]
    
    if robots[leader_index].detected_fire:
        for i, robot in enumerate(robots):
            if robot.is_leader:
                # Enhanced leader behavior
                if not robot.circling:
                    robot.circling = True
                    robot.current_angle = np.arctan2(
                        poses[1, i] - fire_zone.center[1],
                        poses[0, i] - fire_zone.center[0]
                    )
                
                # Slower rotation for better tracking
                robot.current_angle += 0.015  # Slightly faster than before
                
                # Blend optimal position with circling
                optimal_pos = robot.compute_optimal_position(fire_zone.center, poses[:2, i])
                circle_pos = robot.compute_circling_position(fire_zone.center, len(robots))
                targets[:, i] = 0.8 * optimal_pos + 0.2 * circle_pos  # More weight on optimal position
            else:
                if not robot.reached_leader:
                    targets[:, i] = leader_pos
                else:
                    if not robot.circling:
                        robot.circling = True
                        robot.current_angle = 2 * np.pi * i / len(robots)
                    robot.current_angle += 0.02
                    targets[:, i] = robot.compute_circling_position(fire_zone.center, len(robots))
    else:
        # Search pattern
        time_varying = 0.2 * np.sin(poses[2, :])
        search_radius = 0.8 + time_varying
        current_time = 0.02 * iteration_count
        angles = np.linspace(0, 2 * np.pi, len(robots), endpoint=False) + current_time
        targets = np.array([[search_radius[i] * np.cos(angles[i]), 
                           search_radius[i] * np.sin(angles[i])] for i in range(len(robots))]).T
    
    return targets

def main():
    # Initialize Robotarium
    N = 5
    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)
    
    # Create barrier certificate with adjusted parameters for higher speeds
    uni_barrier_cert = create_unicycle_barrier_certificate_with_boundary(safety_radius=0.15)
    si_to_uni_dyn = create_si_to_uni_dynamics(linear_velocity_gain=2.0, angular_velocity_limit=np.pi)
    
    # Initialize fire zone and robots
    fire_zone = FireZone([0.5, 0])
    leader_index = np.random.randint(0, N)
    robots = [Robot(i, is_leader=(i == leader_index)) for i in range(N)]
    
    # Set up plotting
    plt.ion()
    fig = plt.figure(figsize=(15, 6))
    ax_main = fig.add_subplot(121)
    ax_thermal = fig.add_subplot(122)
    
    # Main control loop
    for iteration_count in range(1000):
        poses = r.get_poses()
        
        # Update robots and check for fire detection
        for i, robot in enumerate(robots):
            robot.update_position(poses[:2, i])
            robot.check_reached_leader(poses[:2, i], poses[:2, leader_index])
            
            if robot.is_leader:
                if fire_zone.is_in_zone(poses[:2, i]):
                    robot.detected_fire = True
                if robot.camera:
                    robot.thermal_image = robot.camera.capture(
                        poses[:2, i],
                        poses[2, i],
                        fire_zone.center,
                        fire_zone.radius
                    )
        
        # Compute targets using current poses
        targets = compute_targets(poses, iteration_count, robots, fire_zone, leader_index)
        
        # Calculate desired velocities with improved speed control
        dxi = np.zeros((2, N))
        for i, robot in enumerate(robots):
            direction = targets[:, i] - poses[:2, i]
            norm = np.linalg.norm(direction)
            
            if norm > 0.01:
                # Determine appropriate speed
                if robot.is_leader:
                    # Leader speeds
                    if robot.detected_fire:
                        base_speed = robot.approach_speed
                    else:
                        base_speed = robot.cruise_speed
                else:
                    # Follower speeds
                    if robots[leader_index].detected_fire and not robot.reached_leader:
                        base_speed = robot.approach_speed
                    else:
                        base_speed = robot.cruise_speed
                
                # Add speed variations based on distance
                if norm > 0.5:
                    base_speed *= 1.2
                
                # Add randomness if stalled
                if robot.is_stalled():
                    direction += np.random.normal(0, 0.1, 2)
                    norm = np.linalg.norm(direction)
                
                dxi[:, i] = (direction / norm) * base_speed
            else:
                dxi[:, i] = np.random.normal(0, 0.05, 2)
        
        # Convert to unicycle dynamics and apply barrier certificates
        dxu = si_to_uni_dyn(dxi, poses)
        dxu = uni_barrier_cert(dxu, poses)
        
        # Set velocities
        r.set_velocities(np.arange(N), dxu)
        
        # Visualization
        ax_main.clear()
        ax_thermal.clear()
        
        # Draw fire zone
        fire_circle = plt.Circle(
            fire_zone.center,
            fire_zone.radius,
            color='red',
            alpha=0.3
        )
        ax_main.add_patch(fire_circle)
        
        # Draw heat halo
        halo = plt.Circle(
            fire_zone.center,
            fire_zone.radius * 2,
            color=(1, 0.6, 0),
            alpha=0.1
        )
        ax_main.add_patch(halo)
        
        # Draw robots and sensing ranges
        for i, robot in enumerate(robots):
            color = 'blue' if robot.is_leader else 'gray'
            size = 100 if robot.is_leader else 50
            
            # Draw robot position
            ax_main.scatter(poses[0, i], poses[1, i], c=color, s=size)
            
            # Draw FOV cone for leader
            if robot.is_leader and robot.camera:
                fov_angle = np.deg2rad(robot.camera.fov/2)
                orientation = poses[2, i]
                
                left_angle = orientation - fov_angle
                right_angle = orientation + fov_angle
                view_dist = robot.camera.view_distance
                
                # Draw FOV cone lines
                ax_main.plot([poses[0, i], poses[0, i] + view_dist * np.cos(left_angle)],
                           [poses[1, i], poses[1, i] + view_dist * np.sin(left_angle)],
                           '--', color=color, alpha=0.3)
                ax_main.plot([poses[0, i], poses[0, i] + view_dist * np.cos(right_angle)],
                           [poses[1, i], poses[1, i] + view_dist * np.sin(right_angle)],
                           '--', color=color, alpha=0.3)
                
                # Draw FOV arc
                angles = np.linspace(left_angle, right_angle, 50)
                arc_x = poses[0, i] + view_dist * np.cos(angles)
                arc_y = poses[1, i] + view_dist * np.sin(angles)
                ax_main.plot(arc_x, arc_y, '--', color=color, alpha=0.2)
                
                # Draw detection range circle
                detection_circle = plt.Circle(
                    (poses[0, i], poses[1, i]),
                    view_dist,
                    color=color,
                    fill=False,
                    linestyle=':',
                    alpha=0.2
                )
                ax_main.add_patch(detection_circle)
            
            # Draw velocity vectors and status indicators
            velocity = dxu[:, i]
            speed = np.linalg.norm(velocity)
            
            if robot.is_leader and robot.detected_fire:
                ax_main.plot([poses[0, i], fire_zone.center[0]],
                           [poses[1, i], fire_zone.center[1]],
                           ':', color='red', alpha=0.5)
            
            if speed > 0:
                arrow_length = 0.3
                norm_velocity = velocity / speed
                ax_main.arrow(poses[0, i], poses[1, i], 
                            norm_velocity[0]*arrow_length, norm_velocity[1]*arrow_length,
                            head_width=0.05, head_length=0.08,
                            fc=color, ec=color, alpha=0.7)
            
            if not robot.is_leader and robots[leader_index].detected_fire and not robot.reached_leader:
                ax_main.plot([poses[0, i], poses[0, leader_index]],
                           [poses[1, i], poses[1, leader_index]],
                           '--', color='gray', alpha=0.3)
        
        # Draw thermal camera view
        if robots[leader_index].thermal_image is not None:
            ax_thermal.imshow(robots[leader_index].thermal_image, cmap='hot')
            ax_thermal.set_title('Thermal Image from Leader')
            
            status_text = 'FIRE DETECTED' if robots[leader_index].detected_fire else 'Scanning'
            color = 'red' if robots[leader_index].detected_fire else 'gray'
            ax_thermal.text(0.05, 0.95, status_text,
                          transform=ax_thermal.transAxes, color=color,
                          fontweight='bold',
                          verticalalignment='top')
        
        # Set main plot properties
        ax_main.set_xlim([-1.6, 1.6])
        ax_main.set_ylim([-1.0, 1.0])
        ax_main.set_title('Robotarium: Multi-Agent System')
        ax_main.set_aspect('equal')
        
        # Update display
        plt.draw()
        plt.pause(0.01)
        
        # Step simulation
        r.step()
    
    # Cleanup
    r.call_at_scripts_end()
    plt.close('all')

if __name__ == "__main__":
    main()                          