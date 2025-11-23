import cairo
import random
import math
import sys
import cv2
import numpy as np
import tyro

from dataclasses import dataclass

# Camera transform (position, lookat, roll) - kept for cam_test only
CAMERA_POS = (0.0, 0.5, 0.5)  # Camera position (x, y, z) relative to world origin
CAMERA_LOOKAT = (2.0, 0.0, 0.0)  # Point camera is looking at (x, y, z)
CAMERA_ROLL = 0.0  # Roll angle in radians (rotation around forward axis, 0 = horizontal)

@dataclass
class Config:
    """Unified configuration for all rendering functions."""
    # Command and mode
    command: str = "render"  # cam, glow, canny, or render
    
    # Image dimensions
    width: int = 600
    height: int = 1200
    
    # Object parameters
    num_objects: int = 100
    d_min: float = 0.6
    d_max: float = 4.5
    d_threshold: float = 1.0  # Will be calculated as d_min * 1.25 if not set
    eccentricity_min: float = 1.05
    eccentricity_max: float = 1.25
    
    # Rendering parameters
    blur_amount: float = 0.3
    halo: bool = False
    halo_radius_multiplier: float = 8
    halo_intensity: float = 0.015
    
    # Camera parameters
    fov_h: float = 45.0  # degrees
    
    # Animation parameters
    img: str = "ppt/apple.jpg"  # Input image path
    img_scale: float = 2.0
    fps: int = 10
    init_v: float = 0.3  # Initial velocity
    base_x: float = 2.0
    offset_mode: str = "ray"  # fix, x, or ray
    offset_radius: float = 1.8  # Offset radius for x and ray modes
    offset_ood_prop: float = 0.8
    offset_ood_close_prop: float = 0.3
    offset_radius_ood: float = 4.0  # Offset radius for x and ray modes

    const_speed_prop: float = 0.5
    deceleration_rate: float = 5.0  # Deceleration rate for last 30% (1.0 = normal, >1.0 = faster, <1.0 = slower)
    ax_offset: float = 0.12
    sup: float = 1.0 # Avoid too big
    
    def __post_init__(self):
        """Calculate derived values."""
        self.aspect_ratio = self.height / self.width
        self.fov_h_rad = math.radians(self.fov_h)

def get_color_values(color_type, weight):
    """Calculate halo and core colors based on type and weight."""
    if color_type == "yellow":
        r, g, b = 1.0, 1.0, 1.0
    else:  # blue
        r, g, b = 0.85, 0.9, 1.0
    
    tint = 1.0 * weight
    core_tint = 0.5 * weight
    core_r = r + (1 - r) * (1 - core_tint)
    core_g = g + (1 - g) * (1 - core_tint)
    core_b = b + (1 - b) * (1 - core_tint)
    
    return (r, g, b, tint, core_r, core_g, core_b)

def draw_blurred_circle(cr, x, y, radius, cfg: Config, color_type="yellow", weight=0.5):
    r, g, b, tint, core_r, core_g, core_b = get_color_values(color_type, weight)
    
    # Halo layer with color tint
    if cfg.halo:
        cr.set_operator(cairo.OPERATOR_ADD)
        halo_mask = cairo.RadialGradient(x, y, 0, x, y, radius * cfg.halo_radius_multiplier)
        halo_mask.add_color_stop_rgba(0, r, g, b, cfg.halo_intensity * (1 + tint))
        halo_mask.add_color_stop_rgba(1, r, g, b, 0)
        cr.set_source(halo_mask)
        cr.paint()
    
    # Core layer with subtle color
    cr.set_operator(cairo.OPERATOR_OVER)
    mask = cairo.RadialGradient(x, y, radius * (1 - cfg.blur_amount), x, y, radius)
    mask.add_color_stop_rgba(0, core_r, core_g, core_b, 1)
    mask.add_color_stop_rgba(1, core_r, core_g, core_b, 0)
    cr.set_source(mask)
    cr.paint()

def draw_blurred_ellipse(cr, x, y, radius, eccentricity, rotation, cfg: Config, color_type="yellow", weight=0.5):
    cr.save()
    cr.translate(x, y)
    cr.rotate(rotation)
    cr.scale(1, eccentricity)
    
    r, g, b, tint, core_r, core_g, core_b = get_color_values(color_type, weight)
    
    # Halo layer with color tint
    if cfg.halo:
        cr.set_operator(cairo.OPERATOR_ADD)
        halo_mask = cairo.RadialGradient(0, 0, 0, 0, 0, radius * cfg.halo_radius_multiplier)
        halo_mask.add_color_stop_rgba(0, r, g, b, cfg.halo_intensity * (1 + tint))
        halo_mask.add_color_stop_rgba(1, r, g, b, 0)
        cr.set_source(halo_mask)
        cr.paint()
    
    # Core layer with subtle color
    cr.set_operator(cairo.OPERATOR_OVER)
    mask = cairo.RadialGradient(0, 0, radius * (1 - cfg.blur_amount), 0, 0, radius)
    mask.add_color_stop_rgba(0, core_r, core_g, core_b, 1)
    mask.add_color_stop_rgba(1, core_r, core_g, core_b, 0)
    cr.set_source(mask)
    cr.paint()
    
    cr.restore()

def normalize(v):
    """Normalize a 3D vector."""
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length == 0:
        return (0, 0, 0)
    return (v[0] / length, v[1] / length, v[2] / length)

def cross(a, b):
    """Cross product of two 3D vectors."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]
    )

def dot(a, b):
    """Dot product of two 3D vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

def world_to_camera(x, y, z, camera_pos, camera_lookat, camera_roll=0.0):
    """Transform world coordinates to camera coordinates.
    Returns (cam_x, cam_y, cam_z) in camera space.
    Camera space: +x forward, -y right, +z up."""
    # Step: Translate (subtract camera position)
    px = x - camera_pos[0]
    py = y - camera_pos[1]
    pz = z - camera_pos[2]
    
    # Step: Build camera coordinate system
    # Forward vector: from camera to lookat
    forward = (
        camera_lookat[0] - camera_pos[0],
        camera_lookat[1] - camera_pos[1],
        camera_lookat[2] - camera_pos[2]
    )
    forward = normalize(forward)
    
    # World up vector
    world_up = (0.0, 0.0, 1.0)
    
    # Right vector: forward × world_up
    right = cross(forward, world_up)
    right_len = math.sqrt(right[0]**2 + right[1]**2 + right[2]**2)
    
    # If forward is parallel to world_up, use a different reference
    if right_len < 1e-6:
        # Use world right (1, 0, 0) as reference
        world_right = (1.0, 0.0, 0.0)
        right = cross(forward, world_right)
        right_len = math.sqrt(right[0]**2 + right[1]**2 + right[2]**2)
        if right_len < 1e-6:
            # Forward is along world_right, use world forward (0, 1, 0)
            world_forward = (0.0, 1.0, 0.0)
            right = cross(forward, world_forward)
    
    right = normalize(right)
    
    # Up vector: right × forward (ensures orthogonality)
    up = cross(right, forward)
    up = normalize(up)
    
    # Step: Apply roll rotation (rotate right and up around forward axis)
    cos_roll = math.cos(camera_roll)
    sin_roll = math.sin(camera_roll)
    right_rolled = (
        right[0] * cos_roll - up[0] * sin_roll,
        right[1] * cos_roll - up[1] * sin_roll,
        right[2] * cos_roll - up[2] * sin_roll
    )
    up_rolled = (
        right[0] * sin_roll + up[0] * cos_roll,
        right[1] * sin_roll + up[1] * cos_roll,
        right[2] * sin_roll + up[2] * cos_roll
    )
    
    # Step: Transform point to camera coordinates
    # Camera space: +x forward, -y right, +z up
    cam_x = dot(forward, (px, py, pz))
    cam_y = -dot(right_rolled, (px, py, pz))  # Negative because right is -y in camera space
    cam_z = dot(up_rolled, (px, py, pz))
    
    return (cam_x, cam_y, cam_z)

def project_3d_to_2d(x, y, z, cfg: Config):
    """Project 3D camera coords to 2D pixel coords.
    Camera at origin looking along +x axis. y=left, z=up."""
    if x <= 0:
        return None
    
    # Camera coords: forward=x, right=-y, up=z
    tan_half_fov_h = math.tan(cfg.fov_h_rad / 2)
    tan_half_fov_v = cfg.aspect_ratio * tan_half_fov_h
    
    # Perspective projection to NDC [-1, 1]
    ndc_x = (-y / x) / tan_half_fov_h
    ndc_y = (z / x) / tan_half_fov_v
    
    # NDC to pixel (top-left origin)
    pixel_x = (ndc_x + 1) * cfg.width / 2
    pixel_y = (-ndc_y + 1) * cfg.height / 2
    
    return (pixel_x, pixel_y)

def cam_test(cfg: Config):
    """Test function using old global CAMERA_POS."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, cfg.width, cfg.height)
    cr = cairo.Context(surface)
    
    cr.set_source_rgb(0, 0, 0)
    cr.paint()
    
    # Sample points on square edges at x=1, centered at (1,0,0)
    for _ in range(cfg.num_objects):
        # Step: Generate point attributes first
        d = random.uniform(cfg.d_min, cfg.d_max)
        color_type = "yellow" if random.random() < 0.66 else "blue"
        weight = random.random()
        is_ellipse = d > cfg.d_threshold
        if is_ellipse:
            eccentricity = random.uniform(cfg.eccentricity_min, cfg.eccentricity_max)
            rotation = random.uniform(0, 2 * math.pi)
        else:
            eccentricity = None
            rotation = None
        
        # Step: Generate initial position on square edge
        edge = random.randint(0, 3)
        if edge == 0:  # Top edge
            y, z = random.uniform(-0.5, 0.5), 0.5
        elif edge == 1:  # Bottom edge
            y, z = random.uniform(-0.5, 0.5), -0.5
        elif edge == 2:  # Left edge
            y, z = -0.5, random.uniform(-0.5, 0.5)
        else:  # Right edge
            y, z = 0.5, random.uniform(-0.5, 0.5)
        
        # Step: Move point based on mode
        if cfg.offset_mode == "fix":
            x = 2.0
        elif cfg.offset_mode == "x":
            x = random.uniform(0.5, 3.5)
        elif cfg.offset_mode == "ray":
            # Base position at x=2.0
            base_x = 2.0
            # Calculate direction vector from origin to base position
            direction = math.sqrt(base_x**2 + y**2 + z**2)
            if direction > 0:
                # Normalize direction vector
                dir_x = base_x / direction
                dir_y = y / direction
                dir_z = z / direction
                # Move along ray by random distance between -1.5 and 1.5
                offset = random.uniform(-1.5, 1.5)
                x = base_x + dir_x * offset
                y = y + dir_y * offset
                z = z + dir_z * offset
            else:
                x = base_x
        else:
            raise ValueError(f"Unknown offset_mode: {cfg.offset_mode}")
        
        # Step: Transform to camera coordinates
        cam_x, cam_y, cam_z = world_to_camera(x, y, z, CAMERA_POS, CAMERA_LOOKAT, CAMERA_ROLL)
        
        # Step: Calculate distance from camera (in camera space, camera is at origin)
        distance = math.sqrt(cam_x**2 + cam_y**2 + cam_z**2)
        
        # Step: Project to 2D (using camera coordinates)
        result = project_3d_to_2d(cam_x, cam_y, cam_z, cfg)
        if result:
            px, py = result
            
            # Step: Scale radius based on distance (near-far effect)
            # For ellipses, use larger scale factor
            if is_ellipse:
                radius = (d * eccentricity) / distance
            else:
                radius = d / distance
            
            # Step: Draw the object
            if is_ellipse:
                draw_blurred_ellipse(cr, px, py, radius, eccentricity, rotation, cfg, color_type, weight)
            else:
                draw_blurred_circle(cr, px, py, radius, cfg, color_type, weight)
    
    surface.write_to_png("cam.png")
    print("Rendered to cam.png")

def glow_test(cfg: Config):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, cfg.width, cfg.height)
    cr = cairo.Context(surface)
    
    # Black background
    cr.set_source_rgb(0, 0, 0)
    cr.paint()
    
    # Generate random objects
    for _ in range(cfg.num_objects):
        x = random.uniform(0, cfg.width)
        y = random.uniform(0, cfg.height)
        d = random.uniform(cfg.d_min, cfg.d_max)
        
        radius = d
        color_type = "yellow" if random.random() < 0.66 else "blue"
        weight = random.random()
        
        if d <= cfg.d_threshold:
            draw_blurred_circle(cr, x, y, radius, cfg, color_type, weight)
        else:
            eccentricity = random.uniform(cfg.eccentricity_min, cfg.eccentricity_max)
            rotation = random.uniform(0, 2 * math.pi)
            draw_blurred_ellipse(cr, x, y, radius, eccentricity, rotation, cfg, color_type, weight)
    
    surface.write_to_png("glow.png")
    print("Rendered to glow.png")

def detect_canny_edges(image_path, threshold1=100, threshold2=200):
    """Detect edges using Canny edge detection.
    Returns a list of 2D points (x, y) representing edge pixels."""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    # Extract edge points (non-zero pixels)
    edge_points = []
    height, width = edges.shape
    for y in range(height):
        for x in range(width):
            if edges[y, x] > 0:
                edge_points.append((x, y))
    
    return edge_points

def canny_test(cfg: Config):
    """Read image, detect edges, and draw them on a black background."""
    # Detect edges
    edge_points = detect_canny_edges(cfg.img)
    
    # Read original image to get dimensions
    img = cv2.imread(cfg.img)
    if img is None:
        raise ValueError(f"Could not read image from {cfg.img}")
    
    img_height, img_width = img.shape[:2]
    
    # Create Cairo surface with same dimensions as input image
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, img_width, img_height)
    cr = cairo.Context(surface)
    
    # Black background
    cr.set_source_rgb(0, 0, 0)
    cr.paint()
    
    # Draw edge points in white
    cr.set_source_rgb(1, 1, 1)
    
    for x, y in edge_points:
        cr.rectangle(x, y, 1, 1)
        cr.fill()
    
    # Save to canny.png
    surface.write_to_png("canny.png")
    print(f"Rendered {len(edge_points)} edge points to canny.png")

def get_camera_trajectory(t, cfg):
    """Calculate camera position along trajectory.
    Returns (camera_pos, camera_lookat) tuple."""

    def smooth_lookat_func(x):
        # if z <= 0.5:
        #     y = 4.0 / 3.0 * z * z
        # else:
        #     y = 4.0 / 3.0 * z - 1.0 / 3.0
        return (1.0 - (1.0 - x) ** 3.0) * 0.8 + 0.2

    # Arc trajectory (first 95%)
    # Arc center
    center = (0.0, -1.0, 0.0)
    
    # Start and end points
    start = (-1.0, -1.0, 0.0)
    end = (0.0, 0.0, 0.0)
    
    # Calculate vectors from center
    start_vec = (start[0] - center[0], start[1] - center[1], start[2] - center[2])
    end_vec = (end[0] - center[0], end[1] - center[1], end[2] - center[2])
    
    # Calculate angles
    start_angle = math.atan2(start_vec[1], start_vec[0])
    end_angle = math.atan2(end_vec[1], end_vec[0])
    
    # Interpolate angle (scale t from [0, 0.95] to [0, 1] for arc)

    arc_t1 = 1.0 - t
    ax = arc_t1 * 0.5
    ay = 4 * ax ** 2
    
    # Calculate position on arc
    cam_pos = (
        -ax + cfg.ax_offset,
        -ay,
        (1.0 - t) * 0.5
    )
    
    lookat_t = smooth_lookat_func(t)
    lookat_angle = start_angle + (end_angle - start_angle) * lookat_t
    
    # Calculate lookat direction from lookat_angle
    lookat_direction = (+math.sin(lookat_angle), -math.cos(lookat_angle), 0.0)
    lookat_distance = 1.0
    camera_lookat = (
        cam_pos[0] + lookat_direction[0] * lookat_distance,
        cam_pos[1] + lookat_direction[1] * lookat_distance,
        cam_pos[2] + lookat_direction[2] * lookat_distance
    )

    return cam_pos, camera_lookat

def calculate_arc_length():
    """Calculate total arc length from start to end."""
    center = (0.0, -1.0, 0.0)
    start = (-1.0, -1.0, 0.0)
    end = (0.0, 0.0, 0.0)
    
    start_vec = (start[0] - center[0], start[1] - center[1], start[2] - center[2])
    end_vec = (end[0] - center[0], end[1] - center[1], end[2] - center[2])
    
    start_angle = math.atan2(start_vec[1], start_vec[0])
    end_angle = math.atan2(end_vec[1], end_vec[0])
    
    radius = math.sqrt(start_vec[0]**2 + start_vec[1]**2 + start_vec[2]**2)
    arc_length = radius * abs(end_angle - start_angle)
    
    return arc_length

def get_trajectory_parameter(t, cfg: Config):
    """Calculate next trajectory parameter t based on current t and velocity profile.
    Returns (next_t, should_continue) where should_continue indicates if animation should continue.
    """
    if t >= 1.0:
        return 1.0, False
    
    # Calculate normalized velocity (speed per unit time)
    # Base velocity: init_v normalized by arc length
    arc_length = calculate_arc_length()
    base_velocity = cfg.init_v / arc_length  # Normalized velocity (per second)
    
    # Calculate time step
    dt = 1.0 / cfg.fps
    
    # Velocity profile
    if t < cfg.const_speed_prop:
        # Constant velocity phase (first 70%)
        velocity = base_velocity
    else:
        # Smooth deceleration phase (last 30%)
        # Map t from [0.7, 1.0] to [0, 1] for smoothstep
        u = (t - cfg.const_speed_prop) / (1.0 - cfg.const_speed_prop)  # u in [0, 1] when t in [cfg.const_speed_prop, 1.0]
        
        # Apply deceleration_rate to control decay speed
        # Higher rate (>1.0) = faster decay (steeper curve)
        # Lower rate (<1.0) = slower decay (gentler curve)
        u_adjusted = u ** cfg.deceleration_rate
        
        # Use smoothstep function: s(t) = 3t^2 - 2t^3 for smooth transition
        # Smooth deceleration: velocity decreases smoothly from base_velocity to 0
        # Use (1 - smoothstep) to go from 1 to 0 smoothly
        smoothstep = u_adjusted * u_adjusted * (3.0 - 2.0 * u_adjusted)  # Smoothstep function
        velocity = base_velocity * (1.0 - smoothstep)
    
    # Calculate next t
    next_t = t + velocity * dt
    
    # Clamp to [0, 1] and check if we should continue
    if next_t >= 0.99:
        return 1.0, False
    
    return next_t, True

def render(cfg: Config):
    """Main render function: reads image, extracts edges, samples points, generates animation."""
    # Read image and extract edges
    print(f"Reading image from {cfg.img}")
    edge_points = detect_canny_edges(cfg.img)
    print(f"Detected {len(edge_points)} edge points")
    
    if len(edge_points) == 0:
        raise ValueError("No edge points detected. Check image path and Canny thresholds.")
    
    # Sample points from edges
    if len(edge_points) < cfg.num_objects:
        sampled_points = edge_points
        print(f"Warning: Only {len(edge_points)} edge points available, using all of them")
    else:
        sampled_points = random.sample(edge_points, cfg.num_objects)
    
    # Read original image to get dimensions for coordinate mapping
    img = cv2.imread(cfg.img)
    if img is None:
        raise ValueError(f"Could not read image from {cfg.img}")
    
    img_height, img_width = img.shape[:2]
    
    # Normalize edge points to world coordinates
    # Map image coordinates to world space (e.g., scale to fit in a reasonable range)
    # We'll place points at x=2.0 plane, with y and z mapped from image coordinates
    scale_y = 1.0 / img_width * cfg.img_scale # Scale to [-0.5, 0.5] range
    scale_z = 1.0 / img_height * cfg.img_scale
    
    world_points = []
    for img_x, img_y in sampled_points:
        # Map image coordinates to world coordinates
        # Center at (2.0, 0, 0) with y and z from image
        world_y = -(img_x - img_width / 2) * scale_y
        world_z = -(img_y - img_height / 2) * scale_z
        
        # Apply offset based on offset_mode

        bx = cfg.base_x
        if cfg.offset_mode == "fix":
            x = bx
        elif cfg.offset_mode == "x":
            x = random.uniform(bx - cfg.offset_radius, bx + cfg.offset_radius)
        elif cfg.offset_mode == "ray":
            # Base position at x=2.0
            # Calculate direction vector from origin to base position
            ood = np.random.random() <= cfg.offset_ood_prop
            if ood:
                close = np.random.random() <= cfg.offset_ood_close_prop
                x = bx + random.uniform(-cfg.offset_radius_ood * (0.1 if close else 1.0), 0)
            else:
                direction = math.sqrt(bx**2 + world_y**2 + world_z**2)
                if direction > 0:
                    # Normalize direction vector
                    dir_x = bx / direction
                    dir_y = world_y / direction
                    dir_z = world_z / direction
                    offset = random.uniform(-cfg.offset_radius, cfg.offset_radius)
                    x = bx + dir_x * offset
                    world_y = world_y + dir_y * offset
                    world_z = world_z + dir_z * offset
                else:
                    x = bx 
        else:
            raise ValueError(f"Unknown offset_mode: {cfg.offset_mode}")
        
        world_points.append((x, world_y, world_z))
    
    # Generate object attributes for each point
    objects = []
    for x, y, z in world_points:
        d = random.uniform(cfg.d_min, cfg.d_max)
        color_type = "yellow" if random.random() < 0.66 else "blue"
        weight = random.random()
        is_ellipse = d > cfg.d_threshold
        if is_ellipse:
            eccentricity = random.uniform(cfg.eccentricity_min, cfg.eccentricity_max)
            rotation = random.uniform(0, 2 * math.pi)
        else:
            eccentricity = None
            rotation = None
        
        objects.append({
            'pos': (x, y, z),
            'd': d,
            'color_type': color_type,
            'weight': weight,
            'is_ellipse': is_ellipse,
            'eccentricity': eccentricity,
            'rotation': rotation
        })
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter('render.mp4', fourcc, cfg.fps, (cfg.width, cfg.height))
    
    if not video_writer.isOpened():
        raise RuntimeError("Failed to open video writer")
    
    # Animation loop
    frame_count = 0
    t = 0.0  # Start at beginning of trajectory
    
    print("Rendering frames...")
    
    while True:
        # Calculate trajectory parameter
        t, should_continue = get_trajectory_parameter(t, cfg)
        
        # Get camera position and lookat
        camera_pos, camera_lookat = get_camera_trajectory(t, cfg)
        camera_roll = 0.0
        
        # Render frame
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, cfg.width, cfg.height)
        cr = cairo.Context(surface)
        
        # Black background
        cr.set_source_rgb(0, 0, 0)
        cr.paint()
        
        # Draw objects
        for obj in objects:
            x, y, z = obj['pos']
            
            # Transform to camera coordinates
            cam_x, cam_y, cam_z = world_to_camera(x, y, z, camera_pos, camera_lookat, camera_roll)
            
            # Calculate distance from camera
            distance = math.sqrt(cam_x**2 + cam_y**2 + cam_z**2)
            
            # Project to 2D
            result = project_3d_to_2d(cam_x, cam_y, cam_z, cfg)
            if result:
                px, py = result

                distance = distance if distance >= cfg.sup else cfg.sup - (cfg.sup - distance) * 0.3
                if obj['is_ellipse']:
                    radius = (obj['d'] * obj['eccentricity']) / distance
                    if max(-px, -py, px - cfg.width, py - cfg.height) > radius * cfg.halo_radius_multiplier:
                        continue
                    draw_blurred_ellipse(cr, px, py, radius, obj['eccentricity'], 
                                       obj['rotation'], cfg, obj['color_type'], obj['weight'])
                else:
                    radius = obj['d'] / distance
                    if max(-px, -py, px - cfg.width, py - cfg.height) > radius * cfg.halo_radius_multiplier:
                        continue
                    draw_blurred_circle(cr, px, py, radius, cfg, obj['color_type'], obj['weight'])
        
        # Convert Cairo surface to numpy array for OpenCV
        buf = surface.get_data()
        arr = np.frombuffer(buf, dtype=np.uint8)
        arr = arr.reshape((cfg.height, cfg.width, 4))
        
        # Convert ARGB to BGR for OpenCV
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        # imwrite
        if cfg.fps == 1:
            cv2.imwrite(f"render.png", bgr)
            break
        
        # Write frame
        video_writer.write(bgr)
        
        frame_count += 1
        
        if frame_count % 3 == 0:
            print(f"Rendered frame {frame_count}, t={t:.3f}")
        
        # Check if we should stop
        if not should_continue or t >= 1.0:
            break
    
    video_writer.release()
    print(f"Rendered {frame_count} frames to render.mp4")

def main():
    """Main entry point."""
    cfg: Config = tyro.cli(Config)
    if cfg.command == "cam":
        cam_test(cfg)
    elif cfg.command == "glow":
        glow_test(cfg)
    elif cfg.command == "canny":
        canny_test(cfg)
    elif cfg.command == "render":
        render(cfg)
    else:
        print("Usage: python render.py [cam|glow|canny|render] [--mode fix|x|ray] [--img PATH] [--num_objects N] ...")

if __name__ == "__main__":
    main()
