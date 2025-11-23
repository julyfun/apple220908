from manim import *

class Blur(Scene):
    def construct(self):
        # Constants
        blur = 0.3
        radius = 2.5
        r_inner = radius * (1 - blur)
        r_outer = radius
        
        # Draw circles
        inner_circle = Circle(radius=r_inner, color=BLUE)
        outer_circle = Circle(radius=r_outer, color=RED)
        
        inner_label = MathTex(r"r_{\text{inner}} = r \cdot (1 - \text{blur})", font_size=36)
        inner_label.next_to(inner_circle, DOWN, buff=0.3)
        inner_label.set_color(BLUE)
        
        outer_label = MathTex(r"r_{\text{outer}} = r", font_size=36)
        outer_label.next_to(outer_circle, DOWN, buff=0.8)
        outer_label.set_color(RED)
        
        self.play(Create(inner_circle), Write(inner_label))
        self.wait(0.5)
        self.play(Create(outer_circle), Write(outer_label))
        self.wait(1)
        
        # Show alpha function
        alpha_formula = MathTex(
            r"\alpha(r) = \begin{cases} 1 & r \leq r_{\text{inner}} \\ "
            r"\frac{r_{\text{outer}} - r}{r_{\text{outer}} - r_{\text{inner}}} & r_{\text{inner}} < r < r_{\text{outer}} \\ "
            r"0 & r \geq r_{\text{outer}} \end{cases}",
            font_size=32
        )
        alpha_formula.to_edge(RIGHT, buff=0.5)
        
        self.play(
            inner_circle.animate.shift(LEFT * 2),
            outer_circle.animate.shift(LEFT * 2),
            inner_label.animate.shift(LEFT * 2),
            outer_label.animate.shift(LEFT * 2),
            Write(alpha_formula)
        )
        self.wait(1)
        
        # Animate radius line
        center = inner_circle.get_center()
        
        # Create multiple radial samples
        angles = [0, PI/4, PI/2, 3*PI/4, PI, 5*PI/4, 3*PI/2, 7*PI/4]
        
        for angle in angles[:4]:
            end_point = center + r_outer * np.array([np.cos(angle), np.sin(angle), 0])
            radius_line = Line(center, end_point, color=YELLOW)
            
            # Create dot that moves along radius
            dot = Dot(center, color=YELLOW)
            alpha_text = always_redraw(lambda: DecimalNumber(
                self.get_alpha_value(np.linalg.norm(dot.get_center() - center), r_inner, r_outer),
                num_decimal_places=2
            ).next_to(dot, UP, buff=0.2).scale(0.7))
            
            self.play(Create(radius_line), FadeIn(dot, alpha_text), run_time=0.3)
            self.play(
                dot.animate.move_to(end_point),
                run_time=1.5,
                rate_func=linear
            )
            self.wait(0.3)
            self.play(FadeOut(radius_line, dot, alpha_text), run_time=0.3)
        
        self.wait(1)
        
        # Show final blurred circle effect
        self.play(
            FadeOut(inner_circle, outer_circle, inner_label, outer_label, alpha_formula)
        )
        
        # Create gradient circle (simulated)
        gradient_circles = VGroup()
        num_layers = 30
        for i in range(num_layers):
            r = r_inner + (r_outer - r_inner) * i / num_layers
            alpha = 1 - i / num_layers
            c = Circle(radius=r, color=WHITE, stroke_width=8, stroke_opacity=alpha)
            gradient_circles.add(c)
        
        self.play(FadeIn(gradient_circles))
        self.wait(2)
    
    def get_alpha_value(self, r, r_inner, r_outer):
        if r <= r_inner:
            return 1.0
        elif r >= r_outer:
            return 0.0
        else:
            return (r_outer - r) / (r_outer - r_inner)

class Glow(Scene):
    def construct(self):
        # Constants
        radius = 1.0
        halo_multiplier = 3
        halo_radius = radius * halo_multiplier
        
        # Part 1: Show single star structure
        star_pos = LEFT * 3
        
        # Core layer
        core = self.create_gradient_circle(star_pos, radius, opacity=1.0, color=WHITE)
        core_label = MathTex(r"\text{Core: } r, \alpha_{\text{high}}", font_size=32, color=YELLOW)
        core_label.next_to(star_pos, DOWN, buff=1.5)
        
        self.play(FadeIn(core), Write(core_label))
        self.wait(1)
        
        # Halo layer
        halo = self.create_gradient_circle(star_pos, halo_radius, opacity=0.3, color=BLUE_C)
        halo_label = MathTex(r"\text{Halo: } 3r, \alpha_{\text{low}}", font_size=32, color=BLUE)
        halo_label.next_to(star_pos, DOWN, buff=2.5)
        
        self.play(FadeIn(halo), Write(halo_label))
        self.wait(1.5)
        
        # Show blending formula
        blend_formula = MathTex(
            r"\text{Normal: } C_{\text{out}} = \alpha C_{\text{src}} + (1-\alpha) C_{\text{dst}}",
            font_size=28
        )
        blend_formula.to_edge(UP, buff=0.5)
        
        add_formula = MathTex(
            r"\text{Additive: } C_{\text{out}} = C_{\text{src}} + C_{\text{dst}}",
            font_size=28,
            color=YELLOW
        )
        add_formula.next_to(blend_formula, DOWN, buff=0.3)
        
        self.play(Write(blend_formula))
        self.wait(1)
        self.play(Write(add_formula))
        self.wait(1.5)
        
        # Part 2: Show multiple stars overlapping
        self.play(
            FadeOut(core, halo, core_label, halo_label, blend_formula, add_formula)
        )
        
        # Create three stars with overlapping halos
        positions = [LEFT * 2, ORIGIN, RIGHT * 2]
        stars = VGroup()
        halos = VGroup()
        
        for pos in positions:
            star_core = self.create_gradient_circle(pos, radius * 0.8, opacity=1.0, color=WHITE)
            star_halo = self.create_gradient_circle(pos, halo_radius * 0.8, opacity=0.25, color=BLUE_C)
            stars.add(star_core)
            halos.add(star_halo)
        
        # Show stars one by one
        for i in range(3):
            self.play(FadeIn(halos[i], stars[i]), run_time=0.8)
            self.wait(0.5)
        
        self.wait(1)
        
        # Highlight overlap regions
        overlap_label = MathTex(
            r"\text{Overlap: } \alpha_1 + \alpha_2 > \alpha_1",
            font_size=36,
            color=YELLOW
        )
        overlap_label.to_edge(UP)
        
        # Show brightening in overlap
        bright_regions = VGroup(
            Dot(LEFT * 1, color=YELLOW, radius=0.15),
            Dot(RIGHT * 1, color=YELLOW, radius=0.15)
        )
        
        self.play(Write(overlap_label))
        self.play(FadeIn(bright_regions, scale=2))
        self.wait(1.5)
        
        # Part 3: Show clustering effect
        self.play(
            FadeOut(stars, halos, bright_regions, overlap_label)
        )
        
        cluster_title = MathTex(
            r"\text{Clustering Effect}",
            font_size=40
        )
        cluster_title.to_edge(UP)
        self.play(Write(cluster_title))
        
        # Create dense cluster
        import random
        random.seed(42)
        cluster_stars = VGroup()
        cluster_halos = VGroup()
        
        for _ in range(15):
            angle = random.uniform(0, 2 * PI)
            dist = random.uniform(0, 1.5)
            pos = np.array([dist * np.cos(angle), dist * np.sin(angle), 0])
            
            small_core = self.create_gradient_circle(pos, radius * 0.4, opacity=0.8, color=WHITE)
            small_halo = self.create_gradient_circle(pos, halo_radius * 0.6, opacity=0.15, color=BLUE_C)
            cluster_stars.add(small_core)
            cluster_halos.add(small_halo)
        
        self.play(FadeIn(cluster_halos), run_time=1.5)
        self.play(FadeIn(cluster_stars), run_time=1)
        
        # Show center is brighter
        center_glow = self.create_gradient_circle(ORIGIN, 2, opacity=0.4, color=YELLOW)
        self.play(FadeIn(center_glow, scale=0.5))
        
        self.wait(2)
    
    def create_gradient_circle(self, center, radius, opacity=1.0, color=WHITE):
        """Create a gradient circle to simulate glow."""
        circles = VGroup()
        num_layers = 20
        for i in range(num_layers):
            r = radius * (1 - i / num_layers)
            alpha = opacity * (1 - i / num_layers)
            c = Circle(radius=r, color=color, stroke_width=6, stroke_opacity=alpha, fill_opacity=alpha * 0.3)
            c.move_to(center)
            circles.add(c)
        return circles

class Sampling(ThreeDScene):
    def construct(self):
        # Set up 3D axes
        axes = ThreeDAxes(
            x_range=[-0.5, 2, 0.5],
            y_range=[-1, 1, 0.5],
            z_range=[-1, 1, 0.5],
            x_length=6,
            y_length=4,
            z_length=4,
        )
        
        # Labels
        x_label = MathTex("x", font_size=36).next_to(axes.x_axis, RIGHT)
        y_label = MathTex("y", font_size=36).next_to(axes.y_axis, LEFT)
        z_label = MathTex("z", font_size=36).next_to(axes.z_axis, UP)
        
        # Camera at origin
        camera_dot = Dot3D(point=axes.c2p(0, 0, 0), color=RED, radius=0.1)
        camera_label = MathTex(r"\text{Camera}", font_size=28, color=RED)
        camera_label.next_to(camera_dot, DOWN + LEFT)
        
        # Set camera view
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        
        self.play(Create(axes), Write(x_label), Write(y_label), Write(z_label))
        self.add_fixed_in_frame_mobjects(camera_label)
        self.play(FadeIn(camera_dot), Write(camera_label))
        self.wait(0.5)
        
        # Square at x=1
        square_center = axes.c2p(1, 0, 0)
        vertices = [
            axes.c2p(1, -0.5, 0.5),   # Top-left
            axes.c2p(1, 0.5, 0.5),    # Top-right
            axes.c2p(1, 0.5, -0.5),   # Bottom-right
            axes.c2p(1, -0.5, -0.5),  # Bottom-left
        ]
        
        square = Polygon(*vertices, color=BLUE, stroke_width=3)
        square_label = MathTex(r"\text{Square at } x=1", font_size=32, color=BLUE)
        self.add_fixed_in_frame_mobjects(square_label)
        square_label.to_edge(UP)
        
        self.play(Create(square), Write(square_label))
        self.wait(1)
        
        # Show edges
        edges = [
            Line(vertices[0], vertices[1], color=YELLOW, stroke_width=5),  # Top
            Line(vertices[2], vertices[3], color=YELLOW, stroke_width=5),  # Bottom
            Line(vertices[3], vertices[0], color=YELLOW, stroke_width=5),  # Left
            Line(vertices[1], vertices[2], color=YELLOW, stroke_width=5),  # Right
        ]
        
        edge_formula = MathTex(r"t \sim \mathcal{U}(-0.5, 0.5)", font_size=32)
        self.add_fixed_in_frame_mobjects(edge_formula)
        edge_formula.to_edge(DOWN)
        
        self.play(*[Create(edge) for edge in edges], Write(edge_formula))
        self.wait(1)
        
        # Sample points on edges
        import random
        random.seed(42)
        sample_dots = VGroup()
        
        for _ in range(20):
            edge_idx = random.randint(0, 3)
            t = random.uniform(0, 1)
            
            if edge_idx == 0:  # Top
                point = axes.c2p(1, -0.5 + t, 0.5)
            elif edge_idx == 1:  # Bottom
                point = axes.c2p(1, 0.5 - t, -0.5)
            elif edge_idx == 2:  # Left
                point = axes.c2p(1, -0.5, 0.5 - t)
            else:  # Right
                point = axes.c2p(1, 0.5, -0.5 + t)
            
            dot = Dot3D(point=point, color=WHITE, radius=0.05)
            sample_dots.add(dot)
        
        sample_label = MathTex(r"S = \{ \mathbf{p}_i \mid i = 1, \ldots, N \}", font_size=32)
        self.add_fixed_in_frame_mobjects(sample_label)
        sample_label.move_to(edge_formula)
        
        self.play(
            FadeOut(edge_formula),
            Write(sample_label)
        )
        
        self.play(FadeIn(sample_dots, lag_ratio=0.05), run_time=2)
        self.wait(1)
        
        # Rotate to show structure
        self.begin_ambient_camera_rotation(rate=0.3)
        self.wait(3)
        self.stop_ambient_camera_rotation()
        
        self.wait(1)

class Projection(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes(
            x_range=[-0.5, 3, 0.5],
            y_range=[-1.5, 1.5, 0.5],
            z_range=[-1.5, 1.5, 0.5],
            x_length=7,
            y_length=4,
            z_length=4,
        )
        
        self.set_camera_orientation(phi=70 * DEGREES, theta=-45 * DEGREES)
        
        # Camera at origin
        camera = Dot3D(point=axes.c2p(0, 0, 0), color=RED, radius=0.15)
        camera_label = MathTex(r"\text{Camera}", font_size=24, color=RED)
        self.add_fixed_in_frame_mobjects(camera_label)
        camera_label.next_to(ORIGIN, DOWN + LEFT, buff=0.3)
        
        # Projection plane close to camera
        plane_x = 0.3
        projection_plane = Rectangle(
            width=0.8, height=0.8, 
            color=BLUE, 
            fill_opacity=0.2,
            stroke_width=3
        )
        projection_plane.rotate(PI/2, axis=UP)
        projection_plane.move_to(axes.c2p(plane_x, 0, 0))
        
        self.play(Create(axes), FadeIn(camera), Write(camera_label))
        self.wait(0.3)
        self.play(Create(projection_plane))
        self.wait(0.5)
        
        # Far point
        t_tracker = ValueTracker(0)
        
        def get_far_point():
            t = t_tracker.get_value()
            x = 2.0
            y = 0.6 * np.sin(t * PI)
            z = 0.6 * np.cos(t * PI)
            return axes.c2p(x, y, z)
        
        far_dot = always_redraw(
            lambda: Dot3D(point=get_far_point(), color=YELLOW, radius=0.12)
        )
        
        # Ray extension tracker
        ray_length_tracker = ValueTracker(0)
        camera_pos_3d = axes.c2p(0, 0, 0)
        
        def get_ray_end():
            far_pos = get_far_point()
            camera_coords = axes.p2c(camera_pos_3d)
            far_coords = axes.p2c(far_pos)
            direction = np.array(far_coords) - np.array(camera_coords)
            length = np.linalg.norm(direction)
            if length > 0:
                direction = direction / length
                t = ray_length_tracker.get_value()
                end_coords = np.array(camera_coords) + direction * length * t
                return axes.c2p(*end_coords)
            return far_pos
        
        # Ray from camera to far point (extending)
        ray = always_redraw(
            lambda: Line3D(
                camera_pos_3d,
                get_ray_end(),
                color=WHITE,
                stroke_width=2
            )
        )
        
        # Projected point on plane
        def get_projected_point():
            far_pt = get_far_point()
            far_coords = axes.p2c(far_pt)
            # Perspective projection: scale to plane_x
            scale = plane_x / far_coords[0]
            proj_coords = [plane_x, far_coords[1] * scale, far_coords[2] * scale]
            return axes.c2p(*proj_coords)
        
        projected_dot = always_redraw(
            lambda: Dot3D(point=get_projected_point(), color=GREEN, radius=0.1)
        )
        
        self.play(FadeIn(far_dot))
        self.wait(0.3)
        
        # Animate ray extending from camera to far point
        self.add(ray)
        self.play(ray_length_tracker.animate.set_value(1), run_time=0.8)
        
        # Switch to always follow far point (remove tracker, use direct connection)
        ray = always_redraw(
            lambda: Line3D(
                camera_pos_3d,
                get_far_point(),
                color=WHITE,
                stroke_width=2
            )
        )
        self.remove(ray)
        self.add(ray)
        
        self.wait(0.3)
        self.play(FadeIn(projected_dot))
        self.wait(0.5)
        
        # Move far point
        self.play(t_tracker.animate.set_value(1), run_time=2, rate_func=linear)
        self.wait(0.5)
        
        # Fade out camera for close-up
        self.play(FadeOut(camera, camera_label))
        self.wait(0.3)
        
        # Switch to camera POV: move camera to its actual position and look at (1, 0, 0)
        camera_pos = axes.c2p(0, 0, 0)
        self.move_camera(
            frame_center=(0, 0, 0),
            phi=90 * DEGREES,
            theta=180 * DEGREES,
            zoom=2,
            run_time=2.5
        )
        self.wait(0.3)
        
        # Continue moving point in close-up view
        self.play(t_tracker.animate.set_value(2), run_time=2.5, rate_func=linear)
        self.wait(1)

