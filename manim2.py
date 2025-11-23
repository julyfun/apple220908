from manim import *
import random
import numpy as np
import cv2
random.seed(42)
class CameraMove(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(x_range=[-1, 4, 1], y_range=[-3, 3, 1], z_range=[-3, 3, 1], x_length=8, y_length=6, z_length=6)
        self.set_camera_orientation(phi=70*DEGREES, theta=-45*DEGREES)
        square_x, square_size = 2.0, 2.0
        vertices = [axes.c2p(square_x, -square_size/2, square_size/2), axes.c2p(square_x, square_size/2, square_size/2), axes.c2p(square_x, square_size/2, -square_size/2), axes.c2p(square_x, -square_size/2, -square_size/2)]
        square = Polygon(*vertices, color=BLUE, stroke_width=2)
        points_per_edge = 8
        sample_dots = VGroup()
        for edge_idx in range(4):
            for i in range(points_per_edge):
                t = (i + 0.5) / points_per_edge
                if edge_idx == 0:
                    p = axes.c2p(square_x, -square_size/2 + t*square_size, square_size/2)
                elif edge_idx == 1:
                    p = axes.c2p(square_x, square_size/2, square_size/2 - t*square_size)
                elif edge_idx == 2:
                    p = axes.c2p(square_x, square_size/2 - t*square_size, -square_size/2)
                else:
                    p = axes.c2p(square_x, -square_size/2, -square_size/2 + t*square_size)
                sample_dots.add(Dot3D(point=p, color=YELLOW, radius=0.06))
        cam_tracker = ValueTracker(0)
        def get_cam_pos():
            t = cam_tracker.get_value()
            x = -1 + 2 * t + 0.5 * np.sin(2 * PI * t)
            y = -1.5 + 3 * t + np.sin(3 * PI * t) * 0.8
            z = -1.5 + 3 * t + np.cos(4 * PI * t) * 0.6
            return np.array([x, y, z])
        def get_cam_frame():
            cam_pos = get_cam_pos()
            lookat = np.array([2, 0, 0])
            forward = lookat - cam_pos
            forward_len = np.linalg.norm(forward)
            if forward_len < 1e-6:
                forward = np.array([1, 0, 0])
            else:
                forward = forward / forward_len
            world_up = np.array([0, 0, 1])
            right = np.cross(forward, world_up)
            right_len = np.linalg.norm(right)
            if right_len < 1e-6:
                right = np.cross(forward, np.array([1, 0, 0]))
                right_len = np.linalg.norm(right)
            if right_len > 1e-6:
                right = right / right_len
            up = np.cross(right, forward)
            up = up / np.linalg.norm(up)
            plane_corners = [np.array([1.0, 0.6, 0.4]), np.array([1.0, 0.6, -0.4]), np.array([1.0, -0.6, -0.4]), np.array([1.0, -0.6, 0.4])]
            world_corners = [cam_pos + c[0]*forward + c[1]*right + c[2]*up for c in plane_corners]
            return cam_pos, world_corners
        cam_center_dot = always_redraw(lambda: Dot3D(point=axes.c2p(*get_cam_pos()), color=RED, radius=0.1))
        cam_plane = always_redraw(lambda: Polygon(*[axes.c2p(*p) for p in get_cam_frame()[1]], color=GREEN, fill_opacity=0.3, stroke_width=2))
        cam_lines = always_redraw(lambda: VGroup(*[Line3D(axes.c2p(*get_cam_frame()[0]), axes.c2p(*c), color=GREEN, stroke_width=1, stroke_opacity=0.6) for c in get_cam_frame()[1]]))
        self.play(Create(axes), Create(square), FadeIn(sample_dots, lag_ratio=0.05))
        self.add(cam_center_dot, cam_plane, cam_lines)
        self.play(cam_tracker.animate.set_value(1), run_time=8, rate_func=smooth)
        self.wait(1)

class CannyDemo(Scene):
    def construct(self):
        # Load image
        img_path = "ppt/apple.jpg"
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB for Manim
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Prepare all images
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
        blurred_rgb = np.stack([blurred] * 3, axis=-1)
        
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_magnitude = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)
        gradient_rgb = np.stack([gradient_magnitude] * 3, axis=-1)
        
        edges = cv2.Canny(blurred, 100, 200)
        edges_rgb = np.stack([edges] * 3, axis=-1)
        
        # Create image mobjects
        original_img = ImageMobject(img_rgb).set_height(4)
        blurred_img = ImageMobject(blurred_rgb).set_height(4)
        gradient_img = ImageMobject(gradient_rgb).set_height(4)
        edges_img = ImageMobject(edges_rgb).set_height(4)
        
        # Create labels
        original_label = Text("1. Original Image", font_size=36)
        blurred_label = Text("2. Gaussian Blur (Noise Reduction)", font_size=36)
        gradient_label = Text("3. Gradient Magnitude (Sobel)", font_size=36)
        edges_label = Text("4. Canny Edges (Final Result)", font_size=36, color=YELLOW)
        
        # Position labels
        for label in [original_label, blurred_label, gradient_label, edges_label]:
            label.next_to(original_img, UP, buff=0.3)
        
        # Step 1: Original
        self.play(FadeIn(original_img), Write(original_label))
        self.wait(1)
        
        # Step 2: Blur
        self.play(
            ReplacementTransform(original_img, blurred_img),
            ReplacementTransform(original_label, blurred_label)
        )
        self.wait(1)
        
        # Step 3: Gradient
        self.play(
            ReplacementTransform(blurred_img, gradient_img),
            ReplacementTransform(blurred_label, gradient_label)
        )
        self.wait(1)
        
        # Step 4: Canny edges
        self.play(
            ReplacementTransform(gradient_img, edges_img),
            ReplacementTransform(gradient_label, edges_label)
        )
        self.wait(1.5)
        