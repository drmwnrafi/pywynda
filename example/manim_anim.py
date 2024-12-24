from manim import *

class LorenzAttractor(Scene):
    def construct(self):
        # Simulation Parameters
        sigma = 10.0
        rho = 28.0
        beta = 3.0
        dt = 0.01
        steps = 5000
        init_state = np.array([1.0, 1.0, 1.0])

        # Lorenz System Function
        def lorentz_sys(sigma, rho, beta, state):
            x, y, z = state
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            return np.array([dx, dy, dz])

        # Compute Lorenz Attractor States
        state = init_state
        states = [state]
        for _ in range(steps):
            state = state + lorentz_sys(sigma, rho, beta, state) * dt
            states.append(state)

        states = np.array(states)
        points = [np.array([x, y, z]) for x, y, z in states]

        # Create 3D Axes
        axes = ThreeDAxes(
            x_range=[-30, 30, 10],
            y_range=[-30, 30, 10],
            z_range=[-10, 50, 10],
            x_length=7,
            y_length=7,
            z_length=5,
        )

        # Label Axes
        labels = axes.get_axis_labels(
            x_label=MathTex("X"),
            y_label=MathTex("Y"),
            z_label=MathTex("Z"),
        )

        # Create Trajectory
        trajectory = VMobject()
        trajectory.set_color(RED)
        trajectory.set_opacity(0.8)
        trajectory.set_stroke(width=2)

        # Animation of the Lorenz Attractor
        def update_trajectory(traj, dt):
            traj.add_points_as_corners([axes.coords_to_point(*point) for point in points[:int(dt * 10)]])
            return traj

        self.set_camera_orientation(phi=75 * DEGREES, theta=30 * DEGREES)
        self.add(axes, labels)
        self.play(UpdateFromAlphaFunc(trajectory, update_trajectory), run_time=10, rate_func=linear)
        self.wait()
