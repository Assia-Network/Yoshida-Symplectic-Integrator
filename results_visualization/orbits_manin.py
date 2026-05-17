import numpy as np
from manim import *

class SolarSystemAnimation(ThreeDScene):
    def construct(self):
        raw_data = np.load(r"results/positions.npz")['pos']
        has_z = raw_data.shape[2] == 3
        
        r_left = raw_data[:20000:5].copy() 
        r_right = raw_data[:500000:80].copy()

        sun_pos_left = r_left[:, 0, :].copy()
        sun_pos_right = r_right[:, 0, :].copy()
        for b in range(10):
            r_left[:, b, :] -= sun_pos_left
            r_right[:, b, :] -= sun_pos_right

        body_names = ["Sun", "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune", "Pluto"]
        colors = ["#FFD700", "#1E90FF", "#FF4500", "#32CD32", "#FF8C00", "#8A2BE2", "#00CED1", "#FF00FF", "#7FFF00", "#FFFFFF"]

        self.set_camera_orientation(phi=40 * DEGREES, theta=-90 * DEGREES)

        left_center = LEFT * 3.55
        right_center = RIGHT * 3.55

        scale_left = 1.35    
        scale_right = 0.055  

        time_tracker = ValueTracker(0.0)

        grid_left = NumberPlane(
            x_range=[-2, 2, 0.5], y_range=[-2, 2, 0.5],
            background_line_style={"stroke_opacity": 0.15, "stroke_width": 1},
            faded_line_style={"stroke_opacity": 0.05},
            axis_config={"stroke_opacity": 0.3}
        ).scale(scale_left).move_to(left_center)
        
        grid_right = NumberPlane(
            x_range=[-45, 45, 10], y_range=[-45, 45, 10],
            background_line_style={"stroke_opacity": 0.15, "stroke_width": 1},
            faded_line_style={"stroke_opacity": 0.05},
            axis_config={"stroke_opacity": 0.3}
        ).scale(scale_right).move_to(right_center)
        
        self.add(grid_left, grid_right)

        watermark = Tex(r"Author: Jesus Bautista", font_size=20, color=WHITE).set_opacity(0.4).to_corner(DL)

        left_title = Tex(r"\textbf{Inner System} ($N$-body)", font_size=32).scale(0.85).move_to(LEFT * 3.55 + UP * 3.3)
        right_title = Tex(r"\textbf{Full System} ($N$-body)", font_size=32).scale(0.85).move_to(RIGHT * 3.55 + UP * 3.3)
        
        timer_left_text = Tex("Integration time: ", font_size=24).scale(0.8)
        timer_left_num = DecimalNumber(0, num_decimal_places=1, font_size=24).scale(0.8)
        timer_left_yrs = Tex("Years", font_size=24).scale(0.8)
        timer_left_group = VGroup(timer_left_text, timer_left_num, timer_left_yrs).arrange(RIGHT, buff=0.1)
        timer_left_group.move_to(LEFT * 3.55 + UP * 2.8)  

        timer_right_text = Tex("Integration time: ", font_size=24).scale(0.8)
        timer_right_num = DecimalNumber(0, num_decimal_places=1, font_size=24).scale(0.8)
        timer_right_yrs = Tex("Years", font_size=24).scale(0.8)
        timer_right_group = VGroup(timer_right_text, timer_right_num, timer_right_yrs).arrange(RIGHT, buff=0.1)
        timer_right_group.move_to(RIGHT * 3.55 + UP * 2.8) 

        timer_left_num.add_updater(lambda d: d.set_value(time_tracker.get_value() * 10))
        timer_right_num.add_updater(lambda d: d.set_value(time_tracker.get_value() * 250))
        
        timer_left_group.add_updater(lambda g: g.arrange(RIGHT, buff=0.1).move_to(LEFT * 3.55 + UP * 2.8))
        timer_right_group.add_updater(lambda g: g.arrange(RIGHT, buff=0.1).move_to(RIGHT * 3.55 + UP * 2.8))
        
        labels_list = [Tex(name, font_size=26, color=color).scale(0.6) for name, color in zip(body_names, colors)]
        labels_group = VGroup(*labels_list).arrange(RIGHT, buff=0.35).to_edge(DOWN, buff=0.3)
        
        self.add_fixed_in_frame_mobjects(
            watermark,
            left_title, right_title, 
            timer_left_group,
            timer_right_group,
            labels_group
        )

        def get_pos_updater(body_idx, scale, center_pos, data_array):
            max_idx = len(data_array) - 1
            def updater(mobj):
                exact_val = time_tracker.get_value() * max_idx
                idx_floor = int(np.floor(exact_val))
                idx_ceil = min(idx_floor + 1, max_idx)
                alpha = exact_val - idx_floor
                
                pt_floor = data_array[idx_floor, body_idx]
                pt_ceil = data_array[idx_ceil, body_idx]
                interp_pt = pt_floor * (1 - alpha) + pt_ceil * alpha
                
                z = interp_pt[2] if has_z else 0
                mobj.move_to(center_pos + np.array([interp_pt[0], interp_pt[1], z]) * scale)
            return updater

        def get_smooth_trail_updater(sphere_mobj, body_idx, scale, center_pos, data_array, history_pts):
            max_idx = len(data_array) - 1
            def updater(mobj):
                exact_val = time_tracker.get_value() * max_idx
                curr_idx = int(np.floor(exact_val))
                start_idx = max(0, curr_idx - history_pts)
                
                if curr_idx <= start_idx:
                    mobj.set_points_as_corners([center_pos, center_pos])
                    return
                
                history = data_array[start_idx:curr_idx+1, body_idx]
                points = []
                for pt in history:
                    z = pt[2] if has_z else 0
                    points.append(center_pos + np.array([pt[0], pt[1], z]) * scale)
                
                points.append(sphere_mobj.get_center())
                
                mobj.set_points_as_corners(points)
                mobj.set_style(stroke_color=colors[body_idx], fill_opacity=0)
            return updater

        for i in range(5):
            sphere = Sphere(radius=0.10 if i == 0 else 0.05, color=colors[i])
            sphere.add_updater(get_pos_updater(i, scale_left, left_center, r_left))
            self.add(sphere)
            
            trail = VMobject(stroke_width=1.0)
            trail.add_updater(get_smooth_trail_updater(sphere, i, scale_left, left_center, r_left, history_pts=500))
            self.add(trail)

        for i in range(10):
            sphere = Sphere(radius=0.08 if i == 0 else 0.035, color=colors[i])
            sphere.add_updater(get_pos_updater(i, scale_right, right_center, r_right))
            self.add(sphere)
            
            trail = VMobject(stroke_width=0.8)
            trail.add_updater(get_smooth_trail_updater(sphere, i, scale_right, right_center, r_right, history_pts=400))
            self.add(trail)

        self.play(
            time_tracker.animate.set_value(1.0),
            run_time=10,
            rate_func=linear
        )
        self.wait(0.5)