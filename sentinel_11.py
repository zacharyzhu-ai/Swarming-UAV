import rps.robotarium as robotarium
import numpy as np
from scipy.ndimage import label, measurements
import matplotlib.pyplot as plt
from rps.utilities.transformations import *
from rps.utilities.barrier_certificates import *
from rps.utilities.controllers import *

class FireSimulation:
    """Simulates a fire source with realistic growth and intensity patterns"""
    def __init__(self, initial_location, initial_radius=0.1):
        self.location = np.array(initial_location)
        self.radius = initial_radius
        self.intensity = 1.0
        self.growth_rate = 0.0001  # Very slow growth
        self.intensity_fluctuation = 0.005
        self.min_intensity = 0.7
        self.max_radius = 0.5  # Maximum fire radius
        self.age = 0  # Track fire age
        
    def update(self):
        """Update fire properties for one timestep"""
        self.age += 1
        
        # Growth with decay
        if self.radius < self.max_radius:
            growth_decay = 1.0 - (self.radius / self.max_radius)**2
            self.radius += self.growth_rate * growth_decay
        
        # Realistic intensity fluctuation with time correlation
        time_factor = np.sin(self.age * 0.1) * 0.3  # Slow periodic variation
        random_factor = np.random.uniform(-self.intensity_fluctuation, self.intensity_fluctuation)
        self.intensity = np.clip(
            self.intensity + random_factor + time_factor * self.intensity_fluctuation,
            self.min_intensity,
            1.0
        )
    
    def get_state(self):
        """Return current fire state"""
        return {
            'location': self.location,
            'radius': self.radius,
            'intensity': self.intensity,
            'age': self.age
        }
    
    def get_heat_at_position(self, position):
        """Calculate heat intensity at a given position"""
        distance = np.linalg.norm(position - self.location)
        if distance > self.radius * 2:
            return 0
        # Inverse square law with smoothing
        normalized_distance = distance / (self.radius * 2)
        return self.intensity * (1 - normalized_distance**2)

class ThermalCamera:
    def __init__(self, resolution=(100, 100), fov=90, view_distance=1.5):
        self.resolution = resolution
        self.fov = fov
        self.view_distance = view_distance

    def capture(self, position, orientation, fire_location, fire_radius, fire_intensity=1.0):
        """Generate thermal image using NumPy operations instead of OpenCV"""
        thermal_image = np.zeros(self.resolution)
        angle = orientation
        rel_pos = fire_location - position
        distance = np.linalg.norm(rel_pos)
        
        if distance < self.view_distance:
            rel_angle = np.arctan2(rel_pos[1], rel_pos[0]) - angle
            rel_angle = np.rad2deg(rel_angle)
            
            if abs(rel_angle) < self.fov/2:
                x_img = int((rel_angle + self.fov/2) / self.fov * self.resolution[0])
                y_img = int((1 - distance/self.view_distance) * self.resolution[1])
                intensity = fire_intensity * (1 - 0.5*distance/self.view_distance) * 3.0
                radius = int(fire_radius * (1 - 0.5*distance/self.view_distance) * 75)
                
                # Create coordinate grids for vectorized operations
                y_grid, x_grid = np.mgrid[0:self.resolution[1], 0:self.resolution[0]]
                
                # Calculate distances from each point to center of fire
                dist_from_center = np.sqrt((x_grid - x_img)**2 + (y_grid - y_img)**2)
                
                # Create main fire spot
                circle_mask = dist_from_center <= radius
                thermal_image[circle_mask] = intensity
                
                # Add glow effect
                glow_radius = radius * 2
                glow_mask = (dist_from_center <= glow_radius) & (~circle_mask)
                if np.any(glow_mask):
                    glow_intensity = intensity * 0.5 * (1 - dist_from_center[glow_mask]/glow_radius)
                    thermal_image[glow_mask] = np.maximum(thermal_image[glow_mask], glow_intensity)
        
        return thermal_image


class CVDrone:
    def __init__(self, id, is_leader=False):
        self.id = id
        self.is_leader = is_leader
        self.camera = ThermalCamera()
        self.thermal_image = None
        self.detected_fires = []
        self.target_fire_location = None
        self.circling = False
        self.circle_radius = 0.4
        self.current_angle = 0
        self.current_position = np.zeros(2)
        self.current_orientation = 0
        self.target_persistence = 0
        
    def update_pose(self, position, orientation):
        """Update drone's current pose"""
        self.current_position = position
        self.current_orientation = orientation
        
    def process_thermal_image(self, threshold=0.5):
        """Process thermal image using NumPy/SciPy instead of OpenCV"""
        if self.thermal_image is None:
            return []

        # Simple thresholding
        hot_spots = (self.thermal_image > threshold).astype(np.float32)
        
        # Connected component labeling
        labeled_array, num_features = label(hot_spots)
        
        self.detected_fires = []
        largest_area = 0
        
        if num_features > 0:
            # Get properties of each labeled region
            region_slices = measurements.find_objects(labeled_array)
            for i, region_slice in enumerate(region_slices):
                if region_slice is not None:
                    # Get region properties
                    region = (labeled_array == i + 1)
                    area = np.sum(region)
                    
                    if area > 0:
                        # Calculate centroid
                        y_coords, x_coords = np.nonzero(region)
                        cx = int(np.mean(x_coords))
                        cy = int(np.mean(y_coords))
                        
                        self.detected_fires.append((cx, cy, area))
                        
                        if area > largest_area:
                            largest_area = area
                            # Convert thermal image coordinates to Robotarium coordinates
                            norm_x = (cx / self.camera.resolution[0] - 0.5) * 2 * np.tan(np.deg2rad(self.camera.fov/2))
                            norm_y = (1 - cy / self.camera.resolution[1]) * self.camera.view_distance
                            
                            # Transform to global coordinates
                            cos_theta = np.cos(self.current_orientation)
                            sin_theta = np.sin(self.current_orientation)
                            global_x = self.current_position[0] + norm_y * cos_theta - norm_x * sin_theta
                            global_y = self.current_position[1] + norm_y * sin_theta + norm_x * cos_theta
                            
                            self.target_fire_location = np.array([global_x, global_y])
                            self.target_persistence = 5  # Reset persistence counter
        
        return self.detected_fires


    def compute_circling_position(self, fire_location, angle_offset=0):
        """Compute position for circular formation around fire"""
        angle = self.current_angle + angle_offset
        x = fire_location[0] + self.circle_radius * np.cos(angle)
        y = fire_location[1] + self.circle_radius * np.sin(angle)
        return np.array([x, y])

def main():
    # Initialize Robotarium
    N = 5  # Number of robots
    r = robotarium.Robotarium(number_of_robots=N, show_figure=True, sim_in_real_time=True)
    
    # Create barrier certificate and controller
    uni_barrier_cert = create_unicycle_barrier_certificate_with_boundary()
    si_to_uni_dyn = create_si_to_uni_dynamics(linear_velocity_gain=1, angular_velocity_limit=np.pi/2)
    
    # Initialize fire simulation
    fire_sim = FireSimulation([0.5, 0])
    
    # Initialize drones with CV capabilities
    leader_index = np.random.randint(0, N)
    drones = [CVDrone(i, is_leader=(i == leader_index)) for i in range(N)]
    
    def compute_swarm_targets(fire_location, fire_radius):
        """Compute optimal UAV positions around fire."""
        # Check if any drone has detected fire
        fire_detected = any(drone.detected_fires for drone in drones) or any(drone.target_fire_location is not None for drone in drones)
        
        if fire_detected:
            # If fire is detected, organize circular formation
            targets = np.zeros((2, N))
            
            # Update leader's target if fire detected
            if drones[leader_index].detected_fires:
                drones[leader_index].target_fire_location = fire_location
                
            # Share fire location among all drones
            if drones[leader_index].target_fire_location is not None:
                target_fire = drones[leader_index].target_fire_location
                
                # Set all drones to circling mode
                for drone in drones:
                    if not drone.circling:
                        drone.circling = True
                        # Distribute drones evenly around the circle
                        if drone.is_leader:
                            drone.current_angle = 0
                        else:
                            drone.current_angle = 2 * np.pi * drone.id / N
                
                # Update circling positions
                for i, drone in enumerate(drones):
                    if drone.circling:
                        # Update circling angle
                        drone.current_angle += 0.02  # Speed of rotation
                        targets[:, i] = drone.compute_circling_position(target_fire)
                        
            return targets
        else:
            # Default search pattern when no fire detected
            safe_distance = fire_radius + 0.3
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            return np.array([[fire_location[0] + safe_distance * np.cos(a),
                            fire_location[1] + safe_distance * np.sin(a)] for a in angles]).T
    
    # Set up plotting
    plt.ion()
    fig = plt.figure(figsize=(15, 6))
    ax_main = fig.add_subplot(121)
    ax_thermal = fig.add_subplot(122)
    
    for _ in range(1000):  # Run for 1000 steps
        # Get current robot poses
        poses = r.get_poses()
        
        # Update fire simulation
        fire_sim.update()
        fire_state = fire_sim.get_state()
        
        # Update thermal images for all drones
        for i, drone in enumerate(drones):
            drone.update_pose(poses[:2, i], poses[2, i])
            drone.thermal_image = drone.camera.capture(
                poses[:2, i],
                poses[2, i],
                fire_state['location'],
                fire_state['radius'],
                fire_state['intensity']
            )
            drone.process_thermal_image()
        
        # Compute target positions
        targets = compute_swarm_targets(fire_state['location'], fire_state['radius'])
        
        # Calculate desired velocities
        dxi = np.zeros((2, N))
        for i in range(N):
            direction = targets[:, i] - poses[:2, i]
            norm = np.linalg.norm(direction)
            if norm > 0.01:
                # More conservative speeds with smooth acceleration
                if drones[i].is_leader:
                    base_speed = 0.17
                    if drones[i].target_fire_location is not None:
                        # Faster when heading directly to detected fire
                        target_direction = drones[i].target_fire_location - poses[:2, i]
                        current_heading = np.array([np.cos(poses[2, i]), np.sin(poses[2, i])])
                        heading_alignment = np.dot(target_direction/np.linalg.norm(target_direction), current_heading)
                        # Adjust speed based on alignment with target
                        base_speed *= (0.5 + 0.5 * heading_alignment)
                else:
                    base_speed = 0.08
                
                # Smooth acceleration
                current_speed = np.linalg.norm(dxi[:, i])
                if current_speed > 0:
                    # More gradual acceleration for angular movements
                    angular_change = abs(np.arctan2(direction[1], direction[0]) - poses[2, i])
                    if angular_change > np.pi:
                        angular_change = 2*np.pi - angular_change
                    speed_factor = np.cos(angular_change/2)
                    accel_factor = min(1.0, current_speed/base_speed + 0.05)
                    speed = base_speed * accel_factor * max(0.3, speed_factor)
                else:
                    speed = base_speed * 0.3
                
                dxi[:, i] = (direction / norm) * speed
        
        # Convert to unicycle dynamics and apply barrier certificates
        dxu = si_to_uni_dyn(dxi, poses)
        dxu = uni_barrier_cert(dxu, poses)
        
        # Set velocities
        r.set_velocities(np.arange(N), dxu)
        
        # Update visualization
        ax_main.clear()
        ax_thermal.clear()
        
        # Draw fire with intensity-based visualization
        fire_color = (1, 0, 0, fire_state['intensity'])
        fire_circle = plt.Circle(
            fire_state['location'],
            fire_state['radius'],
            color=fire_color,
            alpha=fire_state['intensity']
        )
        ax_main.add_patch(fire_circle)
        
        # Add heat halo effect
        halo_radius = fire_state['radius'] * 2
        halo = plt.Circle(
            fire_state['location'],
            halo_radius,
            color=(1, 0.6, 0),
            alpha=0.2 * fire_state['intensity']
        )
        ax_main.add_patch(halo)
        
        # Draw robots and their FOV
        for i, drone in enumerate(drones):
            color = 'blue' if drone.is_leader else 'gray'
            size = 100 if drone.is_leader else 50
            ax_main.scatter(poses[0, i], poses[1, i], c=color, s=size)
            
            # Draw field of view cone
            fov_angle = np.deg2rad(drone.camera.fov/2)
            left_angle = poses[2, i] - fov_angle
            right_angle = poses[2, i] + fov_angle
            view_dist = drone.camera.view_distance
            
            ax_main.plot([poses[0, i], poses[0, i] + view_dist * np.cos(left_angle)],
                        [poses[1, i], poses[1, i] + view_dist * np.sin(left_angle)],
                        '--', color=color, alpha=0.3)
            ax_main.plot([poses[0, i], poses[0, i] + view_dist * np.cos(right_angle)],
                        [poses[1, i], poses[1, i] + view_dist * np.sin(right_angle)],
                        '--', color=color, alpha=0.3)
        
        # Set axis limits for main view
        ax_main.set_xlim([-1.6, 1.6])
        ax_main.set_ylim([-1.0, 1.0])
        ax_main.set_title('Robotarium: Multi-Agent UAV System')
        
        # Update thermal view from leader
        if drones[leader_index].thermal_image is not None:
            ax_thermal.imshow(drones[leader_index].thermal_image, cmap='hot')
            ax_thermal.set_title('Thermal Image from Leader Drone')
        
        plt.draw()
        plt.pause(0.01)
        
        # Iterate simulation
        r.step()
    
    # Cleanup
    r.call_at_scripts_end()
    plt.close('all')

if __name__ == "__main__":
    main()