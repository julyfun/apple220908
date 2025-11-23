import cairo
import random
import math
import sys

# Global constants
WIDTH = 600
HEIGHT = 1200
NUM_OBJECTS = 100
D_MIN = 1.2
D_MAX = 3.5
D_THRESHOLD = D_MIN * 1.25
ECCENTRICITY_MIN = 1.05
ECCENTRICITY_MAX = 1.25
BLUR_AMOUNT = 0.3
HALO_RADIUS_MULTIPLIER = 10
HALO_INTENSITY = 0.04

# Camera constants
FOV_H = math.radians(45)
ASPECT_RATIO = HEIGHT / WIDTH

# Camera transform (position, lookat, roll)
CAMERA_POS = (0.0, 2.0, 2.0)  # Camera position (x, y, z) relative to world origin
CAMERA_LOOKAT = (2.0, 0.0, 0.0)  # Point camera is looking at (x, y, z)
CAMERA_ROLL = 0.0  # Roll angle in radians (rotation around forward axis, 0 = horizontal)

def get_color_values(color_type, weight):
    """Calculate halo and core colors based on type and weight."""
    if color_type == "yellow":
        r, g, b = 1.0, 1.0, 1.0
    else:  # blue
        r, g, b = 0.85, 0.9, 1.0
    
    tint = 1.0 * weight
    core_tint = 0.4 * weight
    core_r = r + (1 - r) * (1 - core_tint)
    core_g = g + (1 - g) * (1 - core_tint)
    core_b = b + (1 - b) * (1 - core_tint)
    
    return (r, g, b, tint, core_r, core_g, core_b)

def draw_blurred_circle(cr, x, y, radius, color_type="yellow", weight=0.5, blur=BLUR_AMOUNT):
    r, g, b, tint, core_r, core_g, core_b = get_color_values(color_type, weight)
    
    # Halo layer with color tint
    cr.set_operator(cairo.OPERATOR_ADD)
    halo_mask = cairo.RadialGradient(x, y, 0, x, y, radius * HALO_RADIUS_MULTIPLIER)
    halo_mask.add_color_stop_rgba(0, r, g, b, HALO_INTENSITY * (1 + tint))
    halo_mask.add_color_stop_rgba(1, r, g, b, 0)
    cr.set_source(halo_mask)
    cr.paint()
    
    # Core layer with subtle color
    cr.set_operator(cairo.OPERATOR_OVER)
    mask = cairo.RadialGradient(x, y, radius * (1 - blur), x, y, radius)
    mask.add_color_stop_rgba(0, core_r, core_g, core_b, 1)
    mask.add_color_stop_rgba(1, core_r, core_g, core_b, 0)
    cr.set_source(mask)
    cr.paint()

def draw_blurred_ellipse(cr, x, y, radius, eccentricity, rotation, color_type="yellow", weight=0.5, blur=BLUR_AMOUNT):
    cr.save()
    cr.translate(x, y)
    cr.rotate(rotation)
    cr.scale(1, eccentricity)
    
    r, g, b, tint, core_r, core_g, core_b = get_color_values(color_type, weight)
    
    # Halo layer with color tint
    cr.set_operator(cairo.OPERATOR_ADD)
    halo_mask = cairo.RadialGradient(0, 0, 0, 0, 0, radius * HALO_RADIUS_MULTIPLIER)
    halo_mask.add_color_stop_rgba(0, r, g, b, HALO_INTENSITY * (1 + tint))
    halo_mask.add_color_stop_rgba(1, r, g, b, 0)
    cr.set_source(halo_mask)
    cr.paint()
    
    # Core layer with subtle color
    cr.set_operator(cairo.OPERATOR_OVER)
    mask = cairo.RadialGradient(0, 0, radius * (1 - blur), 0, 0, radius)
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

def world_to_camera(x, y, z):
    """Transform world coordinates to camera coordinates.
    Returns (cam_x, cam_y, cam_z) in camera space.
    Camera space: +x forward, -y right, +z up."""
    # Step: Translate (subtract camera position)
    px = x - CAMERA_POS[0]
    py = y - CAMERA_POS[1]
    pz = z - CAMERA_POS[2]
    
    # Step: Build camera coordinate system
    # Forward vector: from camera to lookat
    forward = (
        CAMERA_LOOKAT[0] - CAMERA_POS[0],
        CAMERA_LOOKAT[1] - CAMERA_POS[1],
        CAMERA_LOOKAT[2] - CAMERA_POS[2]
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
    cos_roll = math.cos(CAMERA_ROLL)
    sin_roll = math.sin(CAMERA_ROLL)
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

def project_3d_to_2d(x, y, z):
    """Project 3D camera coords to 2D pixel coords.
    Camera at origin looking along +x axis. y=left, z=up."""
    if x <= 0:
        return None
    
    # Camera coords: forward=x, right=-y, up=z
    tan_half_fov_h = math.tan(FOV_H / 2)
    tan_half_fov_v = ASPECT_RATIO * tan_half_fov_h
    
    # Perspective projection to NDC [-1, 1]
    ndc_x = (-y / x) / tan_half_fov_h
    ndc_y = (z / x) / tan_half_fov_v
    
    # NDC to pixel (top-left origin)
    pixel_x = (ndc_x + 1) * WIDTH / 2
    pixel_y = (-ndc_y + 1) * HEIGHT / 2
    
    return (pixel_x, pixel_y)

def cam_test(mode="fix"):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    cr = cairo.Context(surface)
    
    cr.set_source_rgb(0, 0, 0)
    cr.paint()
    
    # Sample points on square edges at x=1, centered at (1,0,0)
    for _ in range(NUM_OBJECTS):
        # Step: Generate point attributes first
        d = random.uniform(D_MIN, D_MAX)
        color_type = "yellow" if random.random() < 0.66 else "blue"
        weight = random.random()
        is_ellipse = d > D_THRESHOLD
        if is_ellipse:
            eccentricity = random.uniform(ECCENTRICITY_MIN, ECCENTRICITY_MAX)
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
        if mode == "fix":
            x = 2.0
        elif mode == "x":
            x = random.uniform(0.5, 3.5)
        elif mode == "ray":
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
            raise ValueError(f"Unknown mode: {mode}")
        
        # Step: Transform to camera coordinates
        cam_x, cam_y, cam_z = world_to_camera(x, y, z)
        
        # Step: Calculate distance from camera (in camera space, camera is at origin)
        distance = math.sqrt(cam_x**2 + cam_y**2 + cam_z**2)
        
        # Step: Project to 2D (using camera coordinates)
        result = project_3d_to_2d(cam_x, cam_y, cam_z)
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
                draw_blurred_ellipse(cr, px, py, radius, eccentricity, rotation, color_type, weight)
            else:
                draw_blurred_circle(cr, px, py, radius, color_type, weight)
    
    surface.write_to_png("cam.png")
    print("Rendered to cam.png")

def glow_test():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    cr = cairo.Context(surface)
    
    # Black background
    cr.set_source_rgb(0, 0, 0)
    cr.paint()
    
    # Generate random objects
    for _ in range(NUM_OBJECTS):
        x = random.uniform(0, WIDTH)
        y = random.uniform(0, HEIGHT)
        d = random.uniform(D_MIN, D_MAX)
        
        radius = d
        color_type = "yellow" if random.random() < 0.66 else "blue"
        weight = random.random()
        
        if d <= D_THRESHOLD:
            draw_blurred_circle(cr, x, y, radius, color_type, weight)
        else:
            eccentricity = random.uniform(ECCENTRICITY_MIN, ECCENTRICITY_MAX)
            rotation = random.uniform(0, 2 * math.pi)
            draw_blurred_ellipse(cr, x, y, radius, eccentricity, rotation, color_type, weight)
    
    surface.write_to_png("glow.png")
    print("Rendered to glow.png")

if __name__ == "__main__":
    if sys.argv[1] == "cam":
        mode = sys.argv[2] if len(sys.argv) > 2 else "fix"
        cam_test(mode)
    elif sys.argv[1] == "glow":
        glow_test()
    else:
        print("Usage: python render.py [cam|glow] [fix|x|ray]")
