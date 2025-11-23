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

def project_3d_to_2d(x, y, z):
    """Project 3D world coords to 2D pixel coords.
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

def cam_test():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    cr = cairo.Context(surface)
    
    cr.set_source_rgb(0, 0, 0)
    cr.paint()
    
    # Sample 50 points on square edges at x=1, centered at (1,0,0)
    for _ in range(NUM_OBJECTS):
        edge = random.randint(0, 3)
        if edge == 0:  # Top edge
            y, z = random.uniform(-0.5, 0.5), 0.5
        elif edge == 1:  # Bottom edge
            y, z = random.uniform(-0.5, 0.5), -0.5
        elif edge == 2:  # Left edge
            y, z = -0.5, random.uniform(-0.5, 0.5)
        else:  # Right edge
            y, z = 0.5, random.uniform(-0.5, 0.5)
        x = 2.0
        
        result = project_3d_to_2d(x, y, z)
        if result:
            px, py = result
            d = random.uniform(D_MIN, D_MAX)
            radius = d
            
            color_type = "yellow" if random.random() < 0.66 else "blue"
            weight = random.random()
            
            if d <= D_THRESHOLD:
                draw_blurred_circle(cr, px, py, radius, color_type, weight)
            else:
                eccentricity = random.uniform(ECCENTRICITY_MIN, ECCENTRICITY_MAX)
                rotation = random.uniform(0, 2 * math.pi)
                draw_blurred_ellipse(cr, px, py, radius, eccentricity, rotation, color_type, weight)
    
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
        cam_test()
    elif sys.argv[1] == "glow":
        glow_test()
    else:
        print("Usage: python render.py [cam|glow]")
