# This software uses the community version of the manim animation library for python
# https://github.com/ManimCommunity/manim
# The manim license can be found in manim-license.txt.

import math

from manim import *
import numpy as np
from manim.utils.rate_functions import unit_interval
from sklearn import linear_model
import random
from manim.mobject.geometry.tips import ArrowTriangleFilledTip

np.random.seed(42)

formula_font = 28
big_formula_font = 60

steps___ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 15, 18, 25, 40, 50, 70, 100, 140, 200, 280, 400, 600, 900,
                     1500, 2500, 4000, 6000, 9000, 11000, 11500]


def delayed_rate(delay):
    def rf(t: float) -> float:
        return (t - delay) / (1 - delay)
    return rf


def disturb_xy(xx, yy, variance, only_y=False):
    x_points_ = []
    y_points_ = []
    for x_p, y_p in zip(xx, yy):
        pp = np.array([x_p, y_p])
        if only_y:
            pp = np.random.normal(loc=pp, scale=np.array([0, variance]))
        else:
            pp = np.random.normal(loc=pp, scale=variance)
        x_points_.append(pp[0])
        y_points_.append(pp[1])
    return x_points_, y_points_


def get_animations_with_custom_delay(anims, start_times, end_times):
    assert isinstance(anims, list)
    assert isinstance(start_times, list)
    assert isinstance(end_times, list)
    n = len(anims)
    assert len(start_times) == n
    assert len(end_times) == n

    return [AnimationGroup(anim, run_time=end_, rate_func=delayed_rate(delay=start_/end_))
            for anim, start_, end_ in zip(anims, start_times, end_times)]


class LinearRegressionSimulator:
    def __init__(self, min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b,
                 disturb_coeff, point_removal_mode, variance=None, point_distribution="random"):
        self.min_x = min_x
        self.max_x = max_x
        self.n_points_at_once = n_points_at_once
        self.n_updates = n_updates
        self.n_points_to_update_per_turn = n_points_to_update_per_turn
        self.disturb_coeff = disturb_coeff
        self.x_points = []  # x coords of all points
        self.y_points = []  # y coords of all points
        self.added = []  # indices of added points per turn
        self.removed = []  # indices of removed points per turn
        self.all_removed = set()  # indices of all removed points
        self.a = []
        self.b = []
        self.variance = variance
        self.point_distribution = point_distribution
        self.a_b_update_direction = "increasing"
        self.point_removal_mode = point_removal_mode
        self.update_points(n_points_add=self.n_points_at_once, n_points_remove=0, a=init_a, b=init_b)
        for _ in range(self.n_updates):
            a, b = self.update_a_b(a=self.a[-1], b=self.b[-1])
            self.update_points(n_points_add=self.n_points_to_update_per_turn,
                               n_points_remove=self.n_points_to_update_per_turn, a=a, b=b)

    def update_a_b(self, a, b):
        alpha = np.arctan(a)
        delta = 0.08
        if self.a_b_update_direction == "increasing":
            alpha = alpha + delta
        else:
            alpha = alpha - delta
        a = np.tan(alpha)
        if a > 3:
            self.a_b_update_direction = "decreasing"
        if a < 0.5:
            self.a_b_update_direction = "increasing"
        b = b - 0.1
        return a, b

    def select_points_to_remove(self, n_points_remove, a, b):
        n_points = len(self.x_points)
        if n_points == 0:
            return []
        else:
            if n_points < n_points_remove:
                raise ValueError("Attempted to remove more points than there are currently in existence!")
            x = [x_p if i not in self.all_removed else np.nan for i, x_p in enumerate(self.x_points)]
            y = [y_p if i not in self.all_removed else np.nan for i, y_p in enumerate(self.y_points)]
            if self.point_removal_mode == "worst":
                y_predicted = self.f(np.array(x), a, b)
                dist = np.abs(y_predicted - np.array(y))
                dist[np.isnan(dist)] = -10000
                return np.argsort(dist.flatten())[-n_points_remove:]
            elif self.point_removal_mode == "oldest":
                # Return the oldest points. x and y have nans in the same position, so any of them can be used.
                return np.argwhere(np.logical_not(np.isnan(x))).flatten()[:n_points_remove]
            else:
                raise ValueError("Wrong point_removal_mode!")

    def select_existing_points(self):
        x_existing = [self.x_points[idx] for idx in range(len(self.x_points)) if idx not in self.all_removed]
        y_existing = [self.y_points[idx] for idx in range(len(self.y_points)) if idx not in self.all_removed]
        return x_existing, y_existing

    def update_points(self, n_points_add, n_points_remove, a, b):
        idx_to_remove = list(self.select_points_to_remove(n_points_remove, a, b))
        self.all_removed = self.all_removed.union(set(idx_to_remove))
        self.removed.append(list(idx_to_remove))
        # Generate new points according to the old a and b
        x_points_new = np.reshape(self.generate_x_points(n_points_add), (1, -1))
        variance_ratio = 0.75
        n_low_variance = round(n_points_add * variance_ratio)
        scale1 = 0.3 if self.variance is None else self.variance
        scale2 = 1.5 if self.variance is None else self.variance
        y_points_new_1 = np.random.normal(loc=0, scale=scale1, size=n_low_variance)
        y_points_new_2 = np.random.normal(loc=0, scale=scale2, size=n_points_add - n_low_variance)
        y_points_new = np.reshape(np.concatenate([y_points_new_1, y_points_new_2]), x_points_new.shape)
        points = np.vstack([x_points_new, y_points_new])
        alpha = np.arctan(a)
        rotation_matrix_ = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        points_affine = np.dot(rotation_matrix_, points) + np.array([[0], [b]])
        self.x_points += list(points_affine[0, :])
        self.y_points += list(points_affine[1, :])
        self.added.append(list(range(len(self.x_points) - n_points_add, len(self.x_points))))
        fit_new_ab = False
        if fit_new_ab:
            # Learn the new a and b
            lr = linear_model.LinearRegression()
            x, y = self.select_existing_points()
            lr.fit(np.reshape(np.array(x), (-1, 1)), np.array(y))
            self.a.append(lr.coef_[0])
            self.b.append(lr.intercept_)
        else:
            self.a.append(a)
            self.b.append(b)

    def f(self, x, a, b):
        return a * x + b

    def generate_x_points(self, n_points):
        if self.point_distribution == "random":
            return np.random.uniform(self.min_x, self.max_x, n_points)
        elif self.point_distribution == "uniform":
            return np.linspace(self.min_x, self.max_x, num=n_points)

    def disturb(self, x_points, y_points):
        x_points_ = []
        y_points_ = []
        for x_p, y_p in zip(x_points, y_points):
            points = np.array([x_p, y_p])
            points = np.random.normal(loc=points, scale=self.disturb_coeff)
            x_points_.append(points[0])
            y_points_.append(points[1])
        return x_points_, y_points_

    def get_data(self):
        return self.a, self.b, self.x_points, self.y_points, self.added, self.removed

    def get_limits(self):
        x = np.array(self.x_points)
        y = np.array(self.y_points)
        min_x_lim = np.min(x)
        max_x_lim = np.max(x)
        min_y_lim = np.min(y)
        max_y_lim = np.max(y)
        return min_x_lim, max_x_lim, min_y_lim, max_y_lim


class SimplestFunction(Scene):

    def get_arrays(self, i):
        x_points = np.array([-5, -4, -2, 0, 2, 4, 5])
        y_points = []
        y_points.append(np.array([-1, -1, 0.1, 3, -1.5, 2, -0.8]))
        y_points.append(np.array([0.5, -1.7, 0.5, 1.2, -3, 1.4, 3]))
        y_points.append(np.array([3, 1, -0.3, 3, 2, 0.5, -2]))
        y_points.append(np.array([-1, 1, 2, 0, -2, -0.5, 2.5]))
        return x_points, y_points[i]

    def get_f(self, i):
        x_points, y_points = self.get_arrays(i)
        coeff = np.polyfit(x_points, y_points, deg=6)
        p = np.poly1d(coeff)
        def f(x):
            return p(x)
        return f

    def construct(self):
        x_range = 5
        y_range = 5
        ax = Axes(x_range=[-x_range, x_range, 1], y_range=[-y_range, y_range, 1],
                  x_length=12, y_length=8, tips=True, axis_config={"color": BLUE, "include_ticks": False}) \
            .shift(DOWN * 0.5)
        n_functions = 4
        functions = [self.get_f(i) for i in range(n_functions)]
        f_plots = [ax.plot(f, x_range=[-x_range, x_range]) for f in functions]
        f_plots.append(ax.plot(lambda x: x**2 / 2, x_range=[-x_range, x_range]))
        f_plots.append(ax.plot(lambda x: 2.5 * math.cos(x), x_range=[-x_range, x_range]))
        f_plots.append(ax.plot(lambda x:  x + 0.5, x_range=[-x_range*0.9, x_range*0.9]))
        self.play(Create(ax), run_time=0.5)
        self.play(Write(f_plots[0]), run_time=1)
        transform_time = (11.5 - 2 - 1 - 1 - 0.4) / 6
        for i in range(1, len(f_plots) - 1):
            self.play(Transform(f_plots[0], f_plots[i], rate_func=linear), run_time=transform_time)

        def f1_(x):
            return 2.5 * math.cos(x)

        def f2_(x):
            return x + 0.5

        e = ValueTracker(0)
        f_transient = always_redraw(lambda : ax.plot(lambda x:  e.get_value() * f2_(x) + (1 - e.get_value()) * f1_(x),
                                                     x_range=[-x_range, x_range]))
        self.add(f_transient)
        self.remove(f_plots[0], f_plots[-1])
        self.play(e.animate.set_value(1), run_time=transform_time + 0.7)
        self.wait(2.5)


def get_params():
    min_x = -7
    max_x = 7
    n_points_at_once = 60
    n_updates = 20
    n_points_to_update_per_turn = 30
    init_a = 0.4
    init_b = 0
    disturb_coeff = 1
    point_r = 0.04
    point_color = RED
    point_removal_mode = "oldest"
    return min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
           point_r, point_color, point_removal_mode


class LinearRegression(MovingCameraScene):
    def construct(self):
        min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
        point_r, point_color, point_removal_mode = get_params()

        init_b = 1.2
        n_updates = 50

        lr_simulator = LinearRegressionSimulator(min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn,
                                                 init_a, init_b, disturb_coeff, point_removal_mode)
        a, b, x_points, y_points, added, removed = lr_simulator.get_data()
        min_x_lim, max_x_lim, min_y_lim, max_y_lim = lr_simulator.get_limits()

        ax = Axes(x_range=[min_x_lim - 1, max_x_lim + 1, 1], y_range=[min_y_lim - 1, max_y_lim + 1, 1],
                  x_length=12, y_length=8, tips=True, axis_config={"color": BLUE, "include_ticks": False}) \
            .shift(DOWN * 0.5)
        points = [Circle(radius=point_r, fill_color=point_color, color=point_color, fill_opacity=1, stroke_width=2)
                      .move_to(ax.c2p(x_, y_)) for x_, y_ in zip(x_points, y_points)]
        point_creation_time = 1.6
        time_between_rounds = 0.9
        anims = []
        a = np.array([lr_simulator.a[0]])
        b = np.array([lr_simulator.b[0]])
        range_length = 100
        for i in range(1, len(lr_simulator.a)):
            new_range_a = np.linspace(lr_simulator.a[i-1], lr_simulator.a[i], range_length)
            a = np.hstack([a, new_range_a[1:]])
            new_range_b = np.linspace(lr_simulator.b[i-1], lr_simulator.b[i], range_length)
            b = np.hstack([b, new_range_b[1:]])
        line_progress = ValueTracker(0)

        def get_index(e):
            return math.floor((len(a) - 1) * e)

        line = always_redraw(lambda: ax.plot(lambda x: a[get_index(line_progress.get_value())] * x +
                                                       b[get_index(line_progress.get_value())]).set_color(YELLOW))
        total_time = 0
        creation_anim_starting_times = []
        destruction_anim_starting_times = []
        for t in range(n_updates + 1):
            for _ in added[t]:
                creation_anim_starting_times.append(total_time)
            for _ in removed[t]:
                destruction_anim_starting_times.append(total_time)
            total_time = total_time + time_between_rounds
        for idx in range(len(creation_anim_starting_times)):
            start_create = creation_anim_starting_times[idx]
            end_create = start_create + point_creation_time
            create_anim = FadeIn(points[idx], run_time=end_create,
                                 rate_func=delayed_rate(delay=start_create/end_create))
            if idx < len(destruction_anim_starting_times):
                start_destroy = destruction_anim_starting_times[idx]
                end_destroy = start_destroy + point_creation_time
                destroy_anim = FadeOut(points[idx], run_time=end_destroy - start_destroy)
                color_change_1 = points[idx].animate(run_time=(start_destroy - end_create) / 2).set_color([YELLOW])
                color_change_2 = points[idx].animate(run_time=(start_destroy - end_create) / 2).set_color([PURPLE])
                anims.append(Succession(create_anim, color_change_1, color_change_2, destroy_anim))
            else:
                anims.append(create_anim)
        self.add(ax, line)
        t = 2
        line_animation = Succession(Transform(line_progress, line_progress, run_time=t),
                                    line_progress.animate(run_time=n_updates * time_between_rounds - t,
                                               rate_func=linear).set_value((n_updates * time_between_rounds - t) /
                                                                           n_updates / time_between_rounds))
        self.play(*anims, line_animation)
        self.wait(0.1)


class FittedLine(LinearRegression):
    def construct(self):
        min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
        point_r, point_color, point_removal_mode = get_params()

        n_updates = 0  # Points are static
        init_a = 1
        init_b = 2

        lr_simulator = LinearRegressionSimulator(min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn,
                                                 init_a, init_b, disturb_coeff, point_removal_mode)
        a, b, x_points, y_points, added, removed = lr_simulator.get_data()
        min_x_lim, max_x_lim, min_y_lim, max_y_lim = lr_simulator.get_limits()

        ax = Axes(x_range=[min_x_lim - 1, max_x_lim + 1, 1], y_range=[min_y_lim - 1, max_y_lim + 1, 1],
                  x_length=12, y_length=8, tips=True, axis_config={"color": BLUE, "include_ticks": False}) \
            .shift(DOWN * 0.5)
        points = [Circle(radius=point_r, fill_color=point_color, color=point_color, fill_opacity=1, stroke_width=2)
                      .move_to(ax.c2p(x_, y_)) for x_, y_ in zip(x_points, y_points)]
        point_creation_time = 1
        time_to_create_all_points = 5
        start_times = np.random.random(len(points)) * (time_to_create_all_points - point_creation_time)
        anims = [Succession(Wait(run_time=st), FadeIn(p, run_time=point_creation_time),
                            Wait(run_time=time_to_create_all_points - point_creation_time - st))
                 for p, st in zip(points, start_times)]

        line = ax.plot(lambda x: a[0] * x + b[0]).set_color(YELLOW)

        self.play(Create(ax), run_time=1)
        self.play(*anims)
        self.play(Write(line), run_time=2.3)
        self.wait(3.1)


class OccamsRazor(Scene):
    def construct(self):
        m = 5
        min_x_lim = -m
        max_x_lim = m
        min_y_lim = -m
        max_y_lim = m
        x_length = 6
        y_length = 4.7
        shift_h = 3.5
        ax1 = Axes(x_range=[min_x_lim - 1, max_x_lim + 1, 1], y_range=[min_y_lim - 1, max_y_lim + 1, 1],
                  x_length=x_length, y_length=y_length, tips=True, axis_config={"color": BLUE, "include_ticks": False}) \
            .shift(DOWN * 0.1).shift(LEFT * shift_h)
        ax2 = Axes(x_range=[min_x_lim - 1, max_x_lim + 1, 1], y_range=[min_y_lim - 1, max_y_lim + 1, 1],
                  x_length=x_length, y_length=y_length, tips=True, axis_config={"color": BLUE, "include_ticks": False}) \
            .shift(DOWN * 0.1).shift(RIGHT * shift_h)
        min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
        point_r, point_color, point_removal_mode = get_params()
        n_updates = 0
        init_b = 0.5
        init_a = 0.2
        n_points_at_once = 7
        point_r = 0.07
        scale = 0.7
        min_x = min_x * scale
        max_x = max_x * scale
        lr_simulator = LinearRegressionSimulator(min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn,
                                                 init_a, init_b, disturb_coeff, point_removal_mode, variance=0,
                                                 point_distribution="uniform")

        a, b, x_points, y_points, added, removed = lr_simulator.get_data()
        min_x_lim, max_x_lim, min_y_lim, max_y_lim = lr_simulator.get_limits()
        point_creation_time = 0.3
        time_to_create_all_points = 2
        points = [Circle(radius=point_r, fill_color=point_color, color=point_color, fill_opacity=1, stroke_width=2)
                      .move_to(ax1.c2p(x_, y_)) for x_, y_ in zip(x_points, y_points)]
        start_times = np.linspace(0, time_to_create_all_points - point_creation_time, num=len(points))
        anims = [Succession(Wait(run_time=st), FadeIn(p, run_time=point_creation_time),
                            Wait(run_time=time_to_create_all_points - point_creation_time - st))
                 for p, st in zip(points, start_times)]
        points2 = [Circle(radius=point_r, fill_color=point_color, color=point_color, fill_opacity=1, stroke_width=2)
                      .move_to(ax2.c2p(x_, y_)) for x_, y_ in zip(x_points, y_points)]
        anims2 = [Succession(Wait(run_time=st), FadeIn(p, run_time=point_creation_time),
                            Wait(run_time=time_to_create_all_points - point_creation_time - st))
                 for p, st in zip(points2, start_times)]
        anims = anims + anims2
        x_ = list(x_points)
        y_ = list(y_points)
        l = len(x_)
        for i in range(1, l):
            x_new = (x_[i] + x_[i-1]) / 2
            y_new = (y_[i] + y_[i - 1]) / 2
            x_.append(x_new)
            y_.append(2 * y_new)
        coeff = np.polyfit(np.array(x_), np.array(y_), deg=12)
        p = np.poly1d(coeff)

        def sigma(x):
            return 1 / (1 + math.exp(-x))

        def f(x):
            val = p(x)
            thresh = 3
            sign_ = 1 if val > 0 else -1
            val = abs(val)
            if val <= thresh:
                return val * sign_
            else:
                return sign_ * sigma(thresh + (val - thresh))

        squiggle = ax1.plot(lambda x: f(x)).set_color(YELLOW)
        line = ax2.plot(lambda x: a[0] * x + b[0]).set_color(YELLOW)
        font_size = 40
        label_1 = Text(r'First Hypothesis', font_size=font_size).next_to(ax1, DOWN)
        label_2 = Text(r'Second Hypothesis', font_size=font_size).next_to(ax2, DOWN)
        question = Tex(r'Do you know what this principle is called?', font_size=font_size+10).shift(UP*3).\
            set_color(PURE_RED)

        self.play(Create(ax1), Create(ax2), run_time=1)
        self.play(*anims, run_time=1.5)
        self.play(Write(squiggle), run_time=1.5)
        self.wait(0.5)
        self.play(Write(line), run_time=1.5)
        self.play(Write(label_1), Write(label_2), run_time=1.75)
        self.wait(0.75)
        self.play(Write(question), run_time=2)
        self.wait(5)


class Salary(LinearRegression):
    def construct(self):
        self.camera.frame.set_width(self.camera.frame.get_width() * 1.2)
        min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
        point_r, point_color, point_removal_mode = get_params()

        min_x = 0
        max_x = 50
        a = 1000
        b = 30000
        point_r = point_r * 1.8

        def f(x):
            return a * x + b

        n_points = 12
        x_points = np.random.uniform(min_x, max_x, n_points)
        y_points = [f(x) for x in x_points]
        points = np.array([[x, y] for x, y in zip(x_points, y_points)]).T
        sigma_x = 6
        distorted_points = np.random.randn(*points.shape) * np.array([[sigma_x], [sigma_x * a]]) + points
        distorted_points = distorted_points[:, np.where(np.logical_and(distorted_points[0, :] > 0,
                                                                      distorted_points[1, :] > 20000))[0]]
        x_points = list(distorted_points[0, :])
        y_points = list(distorted_points[1, :])
        smallest_distance = 100000
        idx1 = -1
        idx2 = -1
        for ii in range(len(x_points)):
            for jj in range(ii):
                if abs(x_points[ii] - x_points[jj]) < smallest_distance:
                    smallest_distance = abs(x_points[ii] - x_points[jj])
                    if y_points[ii] < y_points[jj]:
                        idx1, idx2 = ii, jj
                    else:
                        idx1, idx2 = jj, ii
        x_points[idx2] += 10
        left_most = 0
        y_points[left_most] = y_points[left_most] - 12000
        greatest_y_idx = np.argmax(np.array(y_points))
        y_points[greatest_y_idx] -= 7000
        x_arr = np.array(x_points)
        y_arr = np.array(y_points)
        sorted_idx = np.argsort(x_arr)
        x_points = list(x_arr[sorted_idx])
        y_points = list(y_arr[sorted_idx])
        is_above_line = [y > f(x) for x, y in zip(x_points, y_points)]
        shift_val = 10000
        for i in range(len(y_points)):
            if is_above_line[i]:
                y_points[i] += shift_val
            else:
                y_points[i] -= shift_val
        ax = Axes(x_range=[0, 50 + 1, 10], y_range=[0, 80000, 20000],
                  x_length=8, y_length=5.7, tips=True, axis_config={"color": BLUE, "include_ticks": True}) \
            .add_coordinates().shift(UP * 0.8)
        label_font = 35
        coord_labels = ax.get_axis_labels(x_label=Text(r"experience (years)", font_size=label_font),
                                          y_label=Text(r"salary ($)", font_size=label_font))
        points = [Circle(radius=point_r, fill_color=point_color, color=point_color, fill_opacity=1, stroke_width=2)
                      .move_to(ax.c2p(x_, y_)) for x_, y_ in zip(x_points, y_points)]
        point_creation_time = 0.6
        time_to_create_all_points = 4
        start_times = np.random.random(len(points)) * (time_to_create_all_points - point_creation_time)
        anims = [Succession(Wait(run_time=st), FadeIn(p, run_time=point_creation_time),
                            Wait(run_time=time_to_create_all_points - point_creation_time - st))
                 for p, st in zip(points, start_times)]

        line = ax.plot(lambda x: f(x)).set_color(YELLOW)
        long_text_template = TexTemplate(
            tex_compiler="xelatex",
            output_format='.xdv',
        )
        long_text_template.add_to_preamble(r"\usepackage{siunitx}\usepackage{stix2}")
        MathTex.set_default(tex_template=long_text_template)
        Tex.set_default(tex_template=long_text_template)
        font_size = 20
        disclaimer = Tex(
            r'\begin{minipage}{15cm}This data is fictional and is only used to illustrate linear regression. It does not reflect real-world salaries.\end{minipage}',
            font_size=font_size).next_to(ax, DOWN*4)
        n_predict = 8
        x_points_predict = np.linspace(min_x + 5, max_x - 5, num=n_predict)
        y_points_predict = np.zeros_like(x_points_predict)
        predict_color = PURPLE
        points_predict = [Circle(radius=point_r, fill_color=predict_color, color=predict_color, fill_opacity=1,
                                 stroke_width=2).move_to(ax.c2p(x_, y_)) for x_, y_ in zip(list(x_points_predict),
                                                                                           list(y_points_predict))]
        question_marks = [Text("?", font_size=30).set_color(YELLOW).next_to(p, UP*0.6) for p in points_predict]
        move_up_start_times = [i * 0.5 for i in range(len(points_predict))]
        move_up_time = 1.2
        distances = [f(x) for x in x_points_predict]
        paths = [TracedPath(p.get_center, dissipating_time=0.5, stroke_opacity=[0, 1]) for p in points_predict]
        self.add(*paths)
        move_up_anims = [AnimationGroup(Succession(Wait(run_time=st),
                                                   p.animate(run_time=move_up_time).move_to(ax.c2p(x, f(x)))),
                                        Succession(Wait(run_time=st), FadeOut(q, run_time=1)))
                         for p, st, d, x, q in
                         zip(points_predict, move_up_start_times, distances, x_points_predict, question_marks)]
        drop_anims = [p.animate(run_time=move_up_time).move_to(ax.c2p(x, 0))
                 for p, st, d, x in zip(points_predict, move_up_start_times, distances, x_points_predict)]

        def poly_slow(t):
            return t ** 6

        highlights = [Circle(radius=point_r * 1.8, stroke_width=2).set_color(GREEN).move_to(p.get_center())
                      for p in points]

        def get_arrow(p, x, above_line):
            start_ = p.get_center()
            end_ = ax.c2p(x, f(x))
            line = Line(start=start_, end=end_, color=BLUE, buff=0.08).add_tip(tip_length=0.2,
                                                                               tip_shape=ArrowTriangleFilledTip)
            shift_ = 0.06
            if above_line:
                line.shift(DOWN * shift_)
            else:
                line.shift(UP * shift_)
            return line

        arrows = [get_arrow(p, x, above) for p, x, above in zip(points, x_points, is_above_line)]
        arrows_without_left_most = [arrow for idx, arrow in enumerate(arrows) if idx != left_most]
        highlights_without_left_most = [highlight for idx, highlight in enumerate(highlights) if idx != left_most]
        highlight_starting_times = [i * 0.3 for i in range(len(highlights_without_left_most))]
        errors = [abs(f(x) - y) for x, y in zip(x_points, y_points)]
        formula_color = YELLOW

        def format_error(val):
            scale = 100
            return int(val) // scale * scale

        def get_error_position(arr, is_above):
            h_shift = 0.3
            v_shift_ratio = 1/3
            horizontal_shift = LEFT * h_shift if is_above else RIGHT * h_shift
            arrow_start = arr.start
            arrow_end = arr.end
            vertical_pos = arrow_start + (arrow_end - arrow_start) * v_shift_ratio
            return vertical_pos + horizontal_shift

        error_labels = [MathTex(str(format_error(error)), font_size=formula_font).set_color(formula_color).
                            move_to(get_error_position(arr, above)) for error, arr, above in
                        zip(errors, arrows, is_above_line)]
        for error_lab, arr, above, err in zip(error_labels, arrows, is_above_line, errors):
            desired_horizontal_distance = 0.15
            if above:
                shift_ = LEFT * desired_horizontal_distance - LEFT * (error_lab.get_right() - arr.get_center())[0]
                if err >= 10000:
                    shift_ = shift_ + 0.1 * LEFT
            else:
                shift_ = RIGHT * desired_horizontal_distance - RIGHT * (error_lab.get_left() - arr.get_center())[0]
            error_lab.shift(shift_)
        highlight_time = 0.3
        arrow_time = 0.3
        error_label_time = 0.3
        error_strings = [str(format_error(error)) for error in errors]
        formula_strings = []
        for idx, err in enumerate(error_strings):
            formula_strings.append(err)
            formula_strings.append("^2")
            if idx < len(error_strings) - 1:
                formula_strings.append("+")
        formula = MathTex(r'Loss = ', *formula_strings, font_size=formula_font).next_to(ax, DOWN * 4.5).set_color(YELLOW)
        formula_left = formula.get_left()
        first_error_coords = formula[1].get_center()
        errors_inside_formula = [formula[1 + 3 * i] for i in range(len(errors))]
        error_positions_inside_formula = [err.get_center() for err in errors_inside_formula]
        for i, err in enumerate(errors_inside_formula):
            err.move_to(error_labels[i])
        left_most_formula = errors_inside_formula[left_most]
        errors_inside_formula_without_left_most = [err for i, err in enumerate(errors_inside_formula) if i != left_most]
        error_move_starting_times = [i * 0.3 for i in range(len(errors_inside_formula))]
        error_moving_time = 0.8
        plus_writing_time = 0.3
        power_write_time = 0.6
        formula_shift_time = 0.7
        pluses = [formula[3 + 3 * i] for i in range(len(x_points) - 1)]
        pluses.append(MathTex("1").move_to(np.array([100, 100, 0])))
        formula_new = MathTex(r'Loss = ', r'\frac{1}{' + str(len(x_points)) + r'}\Big(', *formula_strings, r'\Big)',
                              font_size=formula_font).set_color(YELLOW)
        formula_new.shift(formula_left - formula_new.get_left())
        formula_shift = formula_new[2].get_center() - first_error_coords
        formula_parts_to_shift = errors_inside_formula + pluses
        powers = [formula_new[3 + 3 * i] for i in range(len(x_points))]
        for p in powers:
            p.set_color(BLUE)
        n_people = MathTex(str(len(x_points)) + r'\text{people}', font_size=formula_font)
        n_people.shift(formula[0].get_left() - n_people.get_left() + DOWN * 0.7)

        ###############################################################################################################x
        self.play(Create(ax), Write(coord_labels), run_time=1)
        self.play(*anims, Write(disclaimer), run_time=5)
        self.wait(1)
        self.wait(2)
        self.play(*[Create(p) for p in points_predict], run_time=2)
        self.wait(3)
        self.wait(0.5)
        self.play(*[Write(q) for q in question_marks], run_time=2)
        self.wait(1.5)
        self.wait(2.8)
        self.play(Write(line), run_time=2)
        self.wait(1.2)
        self.wait(2.2)
        self.play(*move_up_anims)
        self.wait(1.1)
        self.remove(*paths)
        self.wait(2)
        self.play(*drop_anims, rate_func=poly_slow, run_time=1.2)
        self.play(*[FadeIn(q) for q in question_marks], run_time=1)
        self.wait(2.8)
        self.wait(2)
        self.play(Write(highlights[left_most]), run_time=1)
        self.wait(2.5)
        self.play(Write(arrows[left_most]), run_time=1)
        self.play(Write(left_most_formula), run_time=1)
        self.wait(2)
        self.play(
            *[Succession(Wait(run_time=t), Write(highlight, run_time=highlight_time), Write(arrow, run_time=arrow_time),
                         Write(error, run_time=error_label_time))
              for highlight, arrow, error, t in
              zip(highlights_without_left_most, arrows_without_left_most, errors_inside_formula_without_left_most,
                  highlight_starting_times)])
        self.play(Unwrite(disclaimer), run_time=0.3)
        self.play(Write(formula[0]), run_time=0.8)

        self.play(*[Succession(Wait(run_time=t), error.animate(run_time=error_moving_time).move_to(pos),
                               Write(plus, run_time=plus_writing_time)) for error, t, pos, plus in
                    zip(errors_inside_formula, error_move_starting_times, error_positions_inside_formula, pluses)])
        self.play(*[p.animate(run_time=formula_shift_time).shift(formula_shift) for p in formula_parts_to_shift])
        self.play(Write(formula_new[1]), Write(formula_new[-1]), run_time=1)
        self.wait(1.7)
        self.play(*[Write(p, run_time=power_write_time) for p in powers])
        self.wait(1.3)


class FunctionChoices(LinearRegression):
    def construct(self):
        min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
        point_r, point_color, point_removal_mode = get_params()

        n_updates = 0  # Points are static
        init_a = 1.3
        init_b = 0.5
        variance = 0.1
        n_points_at_once = 25

        lr_simulator = LinearRegressionSimulator(min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn,
                                                 init_a, init_b, disturb_coeff, point_removal_mode, variance=variance)
        a, b, x_points, y_points, added, removed = lr_simulator.get_data()
        min_x_lim, max_x_lim, min_y_lim, max_y_lim = lr_simulator.get_limits()

        ax = Axes(x_range=[min_x_lim - 1, max_x_lim + 1, 1], y_range=[min_y_lim + 1, max_y_lim + 1, 1],
                  x_length=12, y_length=6, tips=True, axis_config={"color": BLUE, "include_ticks": False}) \
            .shift(UP * 0.1)

        def get_point(x_, y_):
            return Circle(radius=point_r, fill_color=point_color, color=point_color, fill_opacity=1, stroke_width=2) \
            .move_to(ax.c2p(x_, y_))

        points = [get_point(x_, y_) for x_, y_ in zip(x_points, y_points)]
        line = ax.plot(lambda x: a[0] * x + b[0]).set_color(YELLOW)

        def quadratic(x):
            return x**2

        def trigonometric(x):
            return 2 * math.sin(x)

        def unknown(x):
            return 2 * math.sin(3 * x) * math.log(x ** 2) + 0.2 * (x - 2)**2

        def affine(x, a_affine, b_affine):
            return a_affine * x + b_affine

        y_quadratic = [quadratic(x) for x in x_points]
        x_points_, y_points_ = disturb_xy(x_points, y_quadratic, variance=0.3)
        points_quadratic = [get_point(x, y) for x, y in zip(x_points_, y_points_)]
        line_quadratic = ax.plot(quadratic).set_color(YELLOW)

        y_trigonometric = [trigonometric(x) for x in x_points]
        x_points_, y_points_ = disturb_xy(x_points, y_trigonometric, variance=0.15)
        points_trigonometric = [get_point(x, y) for x, y in zip(x_points_, y_points_)]
        line_trigonometric = ax.plot(trigonometric).set_color(YELLOW)

        y_unknown = [unknown(x) for x in x_points]
        x_points_, y_points_ = disturb_xy(x_points, y_unknown, variance=0.15)
        points_unknown = [get_point(x, y) for x, y in zip(x_points_, y_points_)]
        line_unknown = ax.plot(unknown).set_color(YELLOW)

        a_affine = 0.7
        b_affine = 1.7
        spread = 4.3
        x_points = np.linspace(-spread, spread, 10)
        x_points = list(x_points + 0.1 * np.random.randn(len(x_points)))
        x_points[5] += 0.2
        x_points[6] += 0.2
        y_points_orig = [affine(x, a_affine, b_affine) for x in x_points]
        n_above = 100
        disturbance = None
        while n_above != 5:
            disturbance = np.random.randn(len(x_points))
            sign_ = np.sign(disturbance)
            n_above = np.sum(sign_ > 0)
            disturbance = disturbance + 0.5 * sign_
        y_points = list(np.array(y_points_orig) + disturbance)
        y_points[3] -= 0.2
        y_points[4] += 0.2
        points_affine = [get_point(x_, y_) for x_, y_ in zip(x_points, y_points)]
        line_affine = ax.plot(lambda x: a_affine * x + b_affine).set_color(YELLOW)
        line_affine_b0 = ax.plot(lambda x: affine(x, a_affine=a_affine, b_affine=0)).set_color(YELLOW)
        f = MathTex(r'y = ax', '+ b', font_size=big_formula_font).move_to(np.array([-4, 3, 0])).set_color(YELLOW)
        brace = BraceBetweenPoints(ax.c2p(0, 0), ax.c2p(0, b_affine)).set_color(GREEN).shift(LEFT * 0.15)
        b = MathTex(r'b', font_size=big_formula_font).set_color(YELLOW).next_to(brace, RIGHT * 0.5)

        point_time = 1.3
        pause_time = 0.5
        line_write_time = 1
        time_between_points_and_line = 0.3
        self.play(Create(ax), run_time=0.7)
        self.play(*[FadeIn(p) for p in points], run_time=1.3)
        self.wait(0.7)
        self.play(Write(line), run_time=1.3)
        self.wait(0.7)
        self.play(*[Transform(p, p_q) for p, p_q in zip(points, points_quadratic)], run_time=1.7)
        self.wait(0.3)
        self.play(Transform(line, line_quadratic), run_time=1.3)
        self.play(*[Transform(p, p_q) for p, p_q in zip(points, points_trigonometric)], run_time=1)
        self.play(Transform(line, line_trigonometric), run_time=1)
        self.wait(1.3)
        self.play(*[Transform(p, p_q) for p, p_q in zip(points, points_unknown)], run_time=1.3)
        self.wait(1.3)
        self.play(Transform(line, line_unknown), run_time=1.4)
        self.wait(1)
        self.play(*[FadeOut(p) for p in points], Unwrite(line), *[FadeIn(p) for p in points_affine],
                  run_time=2)
        self.wait(5)
        self.play(Write(line_affine_b0), Write(f[0]), run_time=2)
        self.wait(1.5)
        self.play(Transform(line_affine_b0, line_affine), Write(f[1]), runt_time=2.5)
        self.play(Write(brace), run_time=1)
        self.play(Write(b), runt_time=1)
        self.wait(0.5)

        highlights = [Circle(radius=point_r * 1.8, stroke_width=2).set_color(GREEN).move_to(p.get_center())
                      for p in points_affine]

        def get_arrow(p, x, above_line):
            start_ = p.get_center()
            end_ = ax.c2p(x, affine(x, a_affine, b_affine))
            line = Line(start=start_, end=end_, color=BLUE, buff=0.06).add_tip(tip_length=0.2,
                                                                               tip_shape=ArrowTriangleFilledTip)
            shift_ = 0.03
            if above_line:
                line.shift(DOWN * shift_)
            else:
                line.shift(UP * shift_)
            return line

        is_above_line = [y > affine(x, a_affine, b_affine) for x, y in zip(x_points, y_points)]
        arrows = [get_arrow(p, x, above) for p, x, above in zip(points_affine, x_points, is_above_line)]
        highlight_starting_times = [i * 0.4 for i in range(len(highlights))]
        errors = [abs(affine(x, a_affine, b_affine) - y) for x, y in zip(x_points, y_points)]
        formula_color = YELLOW

        def format_error(val):
            d = 100
            return round(val * d) / d

        def get_error_position(arr, is_above):
            h_shift = 0.3
            v_shift_ratio = 1 / 3
            horizontal_shift = LEFT * h_shift if is_above else RIGHT * h_shift
            arrow_start = arr.start
            arrow_end = arr.end
            vertical_pos = arrow_start + (arrow_end - arrow_start) * v_shift_ratio
            return vertical_pos + horizontal_shift

        error_labels = [MathTex(str(format_error(error)), font_size=formula_font).set_color(formula_color).
                            move_to(get_error_position(arr, above)) for error, arr, above in
                        zip(errors, arrows, is_above_line)]
        for error_lab, arr, above, err in zip(error_labels, arrows, is_above_line, errors):
            desired_horizontal_distance = 0.15
            if above:
                shift_ = LEFT * desired_horizontal_distance - LEFT * (error_lab.get_right() - arr.get_center())[0]
                if err >= 10000:
                    shift_ = shift_ + 0.1 * LEFT
            else:
                shift_ = RIGHT * desired_horizontal_distance - RIGHT * (error_lab.get_left() - arr.get_center())[0]
            error_lab.shift(shift_)
        highlight_time = 0.3
        arrow_time = 0.3
        error_label_time = 0.3
        error_strings = [str(format_error(error)) for error in errors]
        formula_strings = []
        for idx, err in enumerate(error_strings):
            formula_strings.append(err)
            formula_strings.append("^2")
            if idx < len(error_strings) - 1:
                formula_strings.append("+")
        formula = MathTex(r'Loss = ', *formula_strings, font_size=formula_font).next_to(ax, DOWN * 0).shift(DOWN * 3).\
            set_color(YELLOW)
        formula_left = formula.get_left()
        first_error_coords = formula[1].get_center()
        errors_inside_formula = [formula[1 + 3 * i] for i in range(len(errors))]
        error_positions_inside_formula = [err.get_center() for err in errors_inside_formula]
        for i, err in enumerate(errors_inside_formula):
            err.move_to(error_labels[i])
        error_move_starting_times = [i * 0.4 for i in range(len(errors_inside_formula))]
        error_moving_time = 0.8
        plus_writing_time = 0.3
        power_write_time = 0.6
        formula_shift_time = 0.7
        pluses = [formula[3 + 3 * i] for i in range(len(x_points) - 1)]
        pluses.append(MathTex("1").move_to(np.array([100, 100, 0])))
        formula_new = MathTex(r'Loss = ', r'\frac{1}{' + str(len(x_points)) + r'}\Big(', *formula_strings, r'\Big)',
                              font_size=formula_font).set_color(YELLOW)
        formula_new.shift(formula_left - formula_new.get_left())
        formula_shift = formula_new[2].get_center() - first_error_coords
        formula_parts_to_shift = errors_inside_formula + pluses
        powers = [formula_new[3 + 3 * i] for i in range(len(x_points))]
        for p in powers:
            p.set_color(BLUE)
        n_people = MathTex(str(len(x_points)) + r'\text{people}', font_size=formula_font)
        n_people.shift(formula[0].get_left() - n_people.get_left() + DOWN * 0.7)

        highlight_time = 0.2
        arrow_time = 0.2
        error_label_time = 0.2
        highlight_starting_times = [i * 0.15 for i in range(len(highlights))]
        self.wait(0.5)
        self.play(
            *[Succession(Wait(run_time=t), Write(highlight, run_time=highlight_time), Write(arrow, run_time=arrow_time),
                         Write(error, run_time=error_label_time))
              for highlight, arrow, error, t in
              zip(highlights, arrows, errors_inside_formula, highlight_starting_times)])
        self.play(Write(formula[0]))

        error_move_starting_times = [i * 0.2 for i in range(len(errors_inside_formula))]
        error_moving_time = 0.4
        plus_writing_time = 0.2
        power_write_time = 0.5
        formula_shift_time = 0.3

        self.play(*[Succession(Wait(run_time=t), error.animate(run_time=error_moving_time).move_to(pos),
                               Write(plus, run_time=plus_writing_time)) for error, t, pos, plus in
                    zip(errors_inside_formula, error_move_starting_times, error_positions_inside_formula, pluses)])
        self.wait(0.7)
        self.play(*[p.animate(run_time=formula_shift_time).shift(formula_shift) for p in formula_parts_to_shift])
        self.play(Write(formula_new[1]), Write(formula_new[-1]), run_time=1)
        self.wait(0.3)
        self.play(*[Write(p, run_time=power_write_time) for p in powers])
        self.wait(1)


class Multidim(ThreeDScene):
    def construct(self):
        self.camera.exponential_projection = True
        axes_shift_extra = LEFT
        min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
        point_r, point_color, point_removal_mode = get_params()

        n_updates = 0  # Points are static
        init_a = 1.3
        init_b = 0.5
        variance = 0.1
        n_points_at_once = 25
        ax_rotation_angle = 0.35

        lr_simulator = LinearRegressionSimulator(min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn,
                                                 init_a, init_b, disturb_coeff, point_removal_mode, variance=variance)
        a, b, x_points, y_points, added, removed = lr_simulator.get_data()

        ax = ThreeDAxes(x_range=[0, 50 + 1, 10], y_range=[0, 80000, 20000], z_range=[0, 50 + 1, 10],
                  x_length=8, y_length=5.7, z_length=8, tips=True, axis_config={"color": BLUE, "include_ticks": False}) \
            .add_coordinates().shift(UP * 0.3).shift(axes_shift_extra)
        ax1 = ax.submobjects[0]
        ax2 = ax.submobjects[1]
        ax3 = ax.submobjects[2]
        axes_ = [ax1, ax2]
        rotation_axis = ax.c2p(0, 1, 0) - ax.c2p(0, 0, 0)
        rotation_point = ax.c2p(0, 0, 0)
        tick1_3 = ax3.submobjects[2].submobjects[0]
        tick2_3 = ax3.submobjects[2].submobjects[1]
        tick3_3 = ax3.submobjects[2].submobjects[2]
        tick4_3 = ax3.submobjects[2].submobjects[3]
        tick5_3 = ax3.submobjects[2].submobjects[4]
        tick1_template_3 = tick1_3.copy()
        tick2_template_3 = tick2_3.copy()
        tick3_template_3 = tick3_3.copy()
        tick4_template_3 = tick4_3.copy()
        tick5_template_3 = tick5_3.copy()
        tick1_3.set_opacity(0)
        tick2_3.set_opacity(0)
        tick3_3.set_opacity(0)
        tick4_3.set_opacity(0)
        tick5_3.set_opacity(0)
        tick1_template_3.set_opacity(0)
        tick2_template_3.set_opacity(0)
        tick3_template_3.set_opacity(0)
        tick4_template_3.set_opacity(0)
        tick5_template_3.set_opacity(0)
        ax3.rotate(angle=PI * ax_rotation_angle, axis=rotation_axis, about_point=rotation_point)
        min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
        point_r, point_color, point_removal_mode = get_params()
        min_x = 4
        max_x = 40
        a = 1000
        a_y = 300
        b = 30000
        point_r = point_r * 1.8

        def f(x):
            return a * x + b

        def f3d(x, y):
            return a * x + a_y * y + b

        n_points = 8
        x_points = list(np.linspace(min_x, max_x, n_points))
        y_points = [f(x) for x in x_points]
        y_add = [12000, -20000, 20000, -10000, 11000, -13000, 16000, -14000]
        y_points = list(np.array(y_points) + np.array(y_add))
        above = list(np.array(y_add) > 0)
        label_font = 35
        coord_labels = ax.get_axis_labels(x_label=Text(r"experience (years)", font_size=label_font),
                                          y_label=Text(r"salary ($)", font_size=label_font))
        points = [Circle(radius=point_r, fill_color=point_color, color=point_color, fill_opacity=1, stroke_width=2)
                      .move_to(ax.c2p(x_, y_)) for x_, y_ in zip(x_points, y_points)]
        point_creation_time = 0.4
        time_to_create_all_points = 2
        start_times = np.random.random(len(points)) * (time_to_create_all_points - point_creation_time)
        line = ax.plot(lambda x: f(x), x_range=[0, 50]).set_color(YELLOW)
        plane = Surface(
            lambda u, v: ax.c2p(u * (1 + v/400), f3d(u * (1 + v/300), v), v),
            resolution=(10, 10),
            v_range=[0, 50],
            u_range=[0, 50],
        ).set_color(YELLOW).set_opacity(0.35).set_shade_in_3d(True)
        plane.rotate(angle=PI * ax_rotation_angle, axis=rotation_axis, about_point=rotation_point)
        long_text_template = TexTemplate(
            tex_compiler="xelatex",
            output_format='.xdv',
        )
        long_text_template.add_to_preamble(r"\usepackage{siunitx}\usepackage{stix2}")
        MathTex.set_default(tex_template=long_text_template)
        Tex.set_default(tex_template=long_text_template)

        def get_arrow_3d(p, ax, arrow_idx, opacity=None):
            coords = ax.p2c(p.get_center())
            x, z, y = coords
            start_ = p.get_center()
            end_ = ax.c2p(x, f3d(x, y), y)
            above_line = z > f3d(x, y)
            line = Line(start=start_, end=end_, color=BLUE, buff=0.08).add_tip(tip_length=0.2,
                                                                               tip_shape=ArrowTriangleFilledTip)
            shift_ = 0.06
            if above_line:
                line.shift(DOWN * shift_)
            else:
                line.shift(UP * shift_)
            if arrow_idx == 1:
                ratio = 0.2777
                old_start = line.start
                start = line.start
                end = line.end
                line.start = start + (end - start) * (1 - ratio)
                line.set_opacity(opacity)
                buff = 0.145
                line_start = Line(start=old_start, end=line.start, color=BLUE, buff=buff)
                line = VGroup(line, line_start)
            return line

        highlight_starting_times = [i * (2.5 - 0.6) / 7.5 for i in range(len(x_points))]
        highlight_time = 0.3
        arrow_time = 0.3

        rotation_group = VGroup()
        for ax_ in axes_:
            rotation_group.add(ax_)
        label_0_coord = Sphere(radius=0.1).move_to(coord_labels[0].get_center()).set_opacity(0)
        label_1_coord = Sphere(radius=0.1).move_to(coord_labels[1].get_center()).set_opacity(0)
        point_centers = [Sphere(radius=0.1).move_to(p.get_center()).set_opacity(0) for p in points]
        rotation_group.add(label_0_coord)
        rotation_group.add(line)
        for p in point_centers:
            rotation_group.add(p)
        label0 = always_redraw(lambda: coord_labels[0].move_to(label_0_coord))
        label1 = always_redraw(lambda: coord_labels[1].move_to(label_1_coord))
        p1 = always_redraw(lambda: points[0].move_to(point_centers[0]))
        p2 = always_redraw(lambda: points[1].move_to(point_centers[1]))
        p3 = always_redraw(lambda: points[2].move_to(point_centers[2]))
        p4 = always_redraw(lambda: points[3].move_to(point_centers[3]))
        p5 = always_redraw(lambda: points[4].move_to(point_centers[4]))
        p6 = always_redraw(lambda: points[5].move_to(point_centers[5]))
        p7 = always_redraw(lambda: points[6].move_to(point_centers[6]))
        p8 = always_redraw(lambda: points[7].move_to(point_centers[7]))
        points_ = [p1, p2, p3, p4, p5, p6, p7, p8]
        tick1 = ax1.submobjects[2].submobjects[0]
        tick2 = ax1.submobjects[2].submobjects[1]
        tick3 = ax1.submobjects[2].submobjects[2]
        tick4 = ax1.submobjects[2].submobjects[3]
        tick5 = ax1.submobjects[2].submobjects[4]
        tick1_template = tick1.copy()
        tick2_template = tick2.copy()
        tick3_template = tick3.copy()
        tick4_template = tick4.copy()
        tick5_template = tick5.copy()
        tick1_template.set_opacity(1)
        tick2_template.set_opacity(1)
        tick3_template.set_opacity(1)
        tick4_template.set_opacity(1)
        tick5_template.set_opacity(1)
        tick_centers = [Sphere(radius=0.1).move_to(tick.get_center()).set_opacity(0)
                         for tick in ax1.submobjects[2].submobjects]
        for t in tick_centers:
            rotation_group.add(t)
        tick1_ = always_redraw(lambda: tick1_template.move_to(tick_centers[0]))
        tick2_ = always_redraw(lambda: tick2_template.move_to(tick_centers[1]))
        tick3_ = always_redraw(lambda: tick3_template.move_to(tick_centers[2]))
        tick4_ = always_redraw(lambda: tick4_template.move_to(tick_centers[3]))
        tick5_ = always_redraw(lambda: tick5_template.move_to(tick_centers[4]))

        tick1_2 = ax2.submobjects[2].submobjects[0]
        tick2_2 = ax2.submobjects[2].submobjects[1]
        tick3_2 = ax2.submobjects[2].submobjects[2]
        tick1_template_2 = tick1_2.copy()
        tick2_template_2 = tick2_2.copy()
        tick3_template_2 = tick3_2.copy()
        tick1_template_2.set_opacity(1)
        tick2_template_2.set_opacity(1)
        tick3_template_2.set_opacity(1)
        tick_templates_2 = [tick1_template_2, tick2_template_2, tick3_template_2]
        z_label = Text(r"education (years)", font_size=label_font).move_to(np.array([4.5, -2.5, 0])).\
            shift(axes_shift_extra)
        formula = MathTex(r'f(x', r') = ax + ', r'c', font_size=formula_font + 14).move_to(np.array([3.9, -0.3, 0])).\
            set_color(YELLOW)
        formula[0][2].set_color(BLUE)
        formula[1][3].set_color(BLUE)
        formula_new = MathTex(r'f(x', r', y', r') = ax + ', r'by +', r'c', font_size=formula_font + 15).\
            set_color(YELLOW)
        formula_new[1][1].set_color(RED)
        formula_new[3][1].set_color(RED)
        formula_new.shift(formula.get_left() - formula_new.get_left())

        ###############################################################################################################
        ticks_original = [tick1, tick2, tick3, tick4, tick5, tick1_2, tick2_2, tick3_2]
        ticks_new = [tick1_, tick2_, tick3_, tick4_, tick5_] + tick_templates_2
        for t in ticks_original:
            t.set_opacity(0)
        for t in ticks_new:
            t.set_opacity(1)
        self.play(Create(ax1), Create(ax2), Write(label0), Write(label1), Write(formula),
                  *[Write(t) for t in ticks_new], run_time=1)

        anims = [Succession(Wait(run_time=st), FadeIn(p, run_time=point_creation_time),
                            Wait(run_time=time_to_create_all_points - point_creation_time - st))
                 for p, st in zip(points_, start_times)]
        self.play(*anims)
        self.wait(0.3)
        self.play(Write(line), run_time=0.7)
        self.wait(1)
        self.play(*[m.animate.rotate(angle=PI * ax_rotation_angle, axis=rotation_axis, about_point=rotation_point)
                    for m in rotation_group], run_time=3)

        tick1_3 = ax3.submobjects[2].submobjects[0]
        tick2_3 = ax3.submobjects[2].submobjects[1]
        tick3_3 = ax3.submobjects[2].submobjects[2]
        tick4_3 = ax3.submobjects[2].submobjects[3]
        tick5_3 = ax3.submobjects[2].submobjects[4]
        ticks3_old = [tick1_3, tick2_3, tick3_3, tick4_3, tick5_3]
        ticks__ = [tick1_template_3, tick2_template_3, tick3_template_3, tick4_template_3, tick5_template_3]
        for t, c in zip(ticks__, ticks3_old):
            t.move_to(c.get_center()).set_opacity(1)
        self.play(Create(ax3), Write(z_label), *[Write(t) for t in ticks__], run_time=1.5) # 9.5s
        point_y_coords = [10, 15, 26, 12, 15.5, 30, 24, 15.5]
        point_3d_coords = [ax.c2p(x_, y_ + p_y_c * a_y, p_y_c) for x_, y_, p_y_c in zip(x_points, y_points, point_y_coords)]
        paths = [TracedPath(p.get_center, dissipating_time=0.5, stroke_opacity=[0, 1]) for p in point_centers]
        self.add(*paths)
        self.play(*[p.animate.move_to(p_pos) for p, p_pos in zip(point_centers, point_3d_coords)], run_time=3.5)
        translucent = 0.4
        opacities = [1 if ab else translucent for ab in above]
        opacities[1] = 1
        self.wait(2)
        self.play(Create(plane), *[p.animate.set_opacity(op) for p, op in zip(points_, opacities)], run_time=3)
        arrows = [get_arrow_3d(p, ax, arrow_idx, translucent).set_opacity(op) for arrow_idx, (p, op) in
                  enumerate(zip(point_centers, opacities))]
        idx = 1
        arrows[1] = get_arrow_3d(point_centers[idx], ax, idx, translucent)
        highlights = [Circle(radius=point_r * 1.8, stroke_width=2, stroke_opacity=op).set_color(GREEN).move_to(p.get_center())
                           for p, op in zip(point_centers, opacities)]
        self.play(
            *[Succession(Wait(run_time=t), Write(highlight, run_time=highlight_time), Write(arrow, run_time=arrow_time))
              for highlight, arrow, t in
              zip(highlights, arrows, highlight_starting_times)])

        shift_1 = RIGHT * (formula_new[2].get_center() - formula[1].get_center())[0]
        shift_2 = RIGHT * (formula_new[4].get_center() - formula[2].get_center() - shift_1)
        self.wait(0.5)
        self.play(formula[1].animate.shift(shift_1), formula[2].animate.shift(shift_1), run_time=0.7)
        self.wait(0.3)
        self.play(Write(formula_new[1]), run_time=1.5)
        self.wait(0.7)
        self.play(formula[2].animate.shift(shift_2), run_time=0.7)
        self.wait(0.3)
        self.play(Write(formula_new[3]), run_time=1.5)
        self.wait(0.3)


def get_arrow_new(point1, point2, ax=None, color=None, buff=None, add_pointer=None):
    if ax is not None:
        point1 = ax.c2p(point1[0], point1[1], point1[2])
        point2 = ax.c2p(point2[0], point2[1], point2[2])
    color_ = YELLOW if color is None else color
    buff_ = 0.05 if buff is None else buff
    line = Line(start=point1, end=point2, color=color_, buff=buff_)
    if add_pointer:
        line.add_tip(tip_shape=ArrowTriangleFilledTip)
    return line


def align_top(obj, template):
    current_coords = obj.get_center()
    x = current_coords[0]
    y = current_coords[1]
    z = current_coords[2]
    shift = template.get_top()[1] - obj.get_top()[1]
    obj.move_to(np.array([x, y + shift, z]))
    return obj


def align_left(obj, template):
    current_coords = obj.get_center()
    x = current_coords[0]
    y = current_coords[1]
    z = current_coords[2]
    shift = template.get_left()[0] - obj.get_left()[0]
    obj.move_to(np.array([x + shift, y, z]))
    return obj


class Execution(ThreeDScene):
    def __init__(self):
        super().__init__()
        self.st = 1
        self.ll = 8
        self.axes_shift = 2.2
        self.axes_counter_shift = 1
        self.x = 0.2
        self.y = 0.3
        self.r = 0.18
        self.c = 1.1
        self.dist_formula_graph = 7
        self.range = 1.2
        self.coord_range = 2
        self.height = 1.1
        self.horizontal_shift = 3
        self.font_size = 55
        self.step_size = 1
        self.formulas_color = YELLOW
        self.numbers_color = RED
        self.highlight_color = GREEN
        self.axes = None
        self.surf = None
        self.steps = None
        self.grads = None
        self.points = None
        self.x_y_dist = None
        self.truncate_number_count = 0
        self.f_0_position = None
        self.n_steps_to_play = 5

    def extra_rotation(self, m):
        return m

    def adjust_labels(self, label_x, label_y, label_z):
        pass

    def f(self, x, y):
        return self.c * (x ** 2 + y ** 2)

    def df_du(self, x, y):
        return 2 * self.c * x

    def df_dv(self, x, y):
        return 2 * self.c * y

    def get_axes(self):
        ratio_ = 0.9
        ax = ThreeDAxes(x_range=[-self.coord_range * ratio_, self.coord_range * ratio_, self.st],
                        x_length=self.ll * ratio_,
                        y_range=[-self.coord_range * self.height, self.coord_range * self.height, self.st],
                        y_length=self.ll*self.height,
                        z_range=[-self.coord_range * ratio_, self.coord_range * ratio_, self.st],
                        z_length=self.ll * ratio_)
        ax.scale(1.5)
        ax.shift(DOWN * 1.2)
        if hasattr(self, 'graph_scale'):
            ax.scale(self.graph_scale)
        return ax

    def normalize_quadratic(self, u, v):
        vec = np.array([u, v])
        len = np.linalg.norm(vec)
        if len > self.range:
            vec = vec / len * self.range
        u = vec[0]
        v = vec[1]
        return u, v

    def get_surf(self, ax, res=None):
        opacity = 1
        if res is None:
            res = 20
        u_range = [-self.range, self.range]
        v_range = [-self.range, self.range]
        return Surface(
            lambda u, v: ax.
                c2p(self.normalize_quadratic(u, v)[0], self.f(self.normalize_quadratic(u, v)[0],
                                                              self.normalize_quadratic(u, v)[1]),
                    self.normalize_quadratic(u, v)[1]),
            u_range=u_range, v_range=v_range, checkerboard_colors=[GREEN_D, GREEN_E],
            resolution=(res, res)).set_opacity(opacity)

    def grad_descent(self, x, y):
        n_iters = 12
        step_size = self.step_size
        self.steps = []
        self.grads = []
        for iter in range(n_iters):
            dx = self.df_du(x, y)
            dy = self.df_dv(x, y)
            self.steps.append([x, y])
            grad = np.array([dx, dy])
            self.grads.append(grad)
            x = x - step_size * grad[0]
            y = y - step_size * grad[1]
        self.steps.append([x, y])

    def xy_to_3d(self, two_d_vec):
        x = two_d_vec[0]
        y = two_d_vec[1]
        return x, self.f(x, y), y

    def axes_graph_setup(self, angle_y_axis=0, other_scene=None, other_coords=None):
        x = self.x
        y = self.y
        self.grad_descent(x, y)
        self.axes = self.get_axes()
        alpha = 1
        secondary_rotation_axis = alpha * RIGHT + (1 - alpha) * OUT
        if other_coords is None:
            self.axes.shift(DOWN * self.axes_shift + LEFT * self.horizontal_shift).rotate(angle=PI / 3, axis=UP) \
                .rotate(angle=angle_y_axis, axis=UP).rotate(angle=PI/36, axis=secondary_rotation_axis) \
                .shift(0.7 * UP)
            self.axes = self.extra_rotation(self.axes)
            self.axes.shift(1.1 * DOWN)
        else:
            self.axes.rotate(axis=IN, angle=-PI/150)
            self.axes.move_to(other_coords)

        axis_label_color = BLUE
        lab_font_size = 46
        x_label = MathTex(r'a', font_size=lab_font_size, color=axis_label_color).\
            move_to(np.array([0.3, -2.6, 0]))
        y_label = MathTex(r'b', font_size=lab_font_size, color=axis_label_color).\
            move_to(np.array([-1, -1.5, 0]))
        z_label = self.get_z_label(font_size=lab_font_size, color=axis_label_color)
        print("z_label initial: " + str(z_label.get_center()))
        self.adjust_labels(x_label, y_label, z_label)
        labels = VGroup(x_label, y_label, z_label)

        self.surf = self.get_surf(self.axes)

        formula_f = MathTex('', font_size=self.font_size)
        if other_scene is None:
            self.play(Create(self.axes), Create(self.surf), Write(x_label), Write(y_label),
                      Write(z_label))
        else:
            other_scene.play(Create(self.axes), Create(self.surf))
        return formula_f

    def get_init_point_helper(self):
        i = 0
        vec = self.steps[i]
        u = vec[0]
        v = vec[1]
        return self.axes.c2p(*self.invert_x_y(np.array([u, self.f(u, v), v])))

    def get_init_point(self):
        print("get_init_point 2")
        prev_point = self.get_init_point_helper()
        return Sphere(radius=self.r).set_color(YELLOW).move_to(prev_point)

    def get_updates(self):
        return MathTex(r'x &:= x - \frac{\partial f}{\partial x}\\',
                          r'y &:= y - \frac{\partial f}{\partial y}',
                          font_size=self.font_size).set_color(self.formulas_color)

    def invert_x_y(self, vec):
        return np.array([vec[2], vec[1], vec[0]])

    def get_steps_for_plotting(self, n_steps):
        steps = VGroup()
        for idx in range(n_steps):
            prev_point = self.steps[idx]
            print("x: " + str(prev_point[0]) + ", y: " + str(prev_point[1]))
            currect_point = self.steps[idx+1]
            color = RED if (np.array(currect_point) - np.array(prev_point))[0] > 0 else BLUE
            new_step = Line(start=self.axes.c2p(*self.invert_x_y(self.xy_to_3d(prev_point))),
                            end=self.axes.c2p(*self.invert_x_y(self.xy_to_3d(currect_point))), color=color, buff=0.05).\
                add_tip(tip_shape=ArrowTriangleFilledTip, tip_length=0.2)
            steps.add(new_step)
        return steps

    def set_color(self, m):
        m[0].set_color(self.formulas_color)
        m[1].set_color(self.numbers_color)
        m[2].set_color(self.formulas_color)
        m[3].set_color(self.numbers_color)
        m[4].set_color(self.formulas_color)
        m[5].set_color(self.formulas_color)
        m[6].set_color(self.numbers_color)
        m[7].set_color(self.formulas_color)
        m[8].set_color(self.formulas_color)
        m[9].set_color(self.numbers_color)
        return m

    def position_xy(self, m):
        current_dist = m[2].get_left()[0] - m[0].get_left()[0]
        coords = m[2].get_center()
        m[2].move_to(np.array([coords[0] + self.x_y_dist - current_dist, coords[1], coords[2]]))

        dist = m[1].get_left()[0] - m[0].get_right()[0]
        dist_current = m[3].get_left()[0] - m[2].get_right()[0]
        m[3].shift(RIGHT * (dist - dist_current))
        return m

    def setup_static_stuff(self):
        formula_f = self.axes_graph_setup()

        formulae_init = MathTex(r'x &=',
                           str('{:2.2f}'.format(self.x)),
                           r'\,\,\,\,' + r'y =',
                           str('{:2.2f}'.format(self.y)) + r'\\',
                           r'\frac{\partial f}{\partial x} &= ' + str(2*self.c) + r'x',
                           r'=',
                           str('{:2.2f}'.format(2 * self.c * self.x)) + r'\\',
                           r'\frac{\partial f}{\partial y} &= ' + str(2*self.c) + r'y',
                           r'=',
                           str('{:2.2f}'.format(2 * self.c * self.y)),
                           font_size=self.font_size).shift(RIGHT * self.horizontal_shift)
        align_top(formulae_init, formula_f)
        formulae_init.shift(RIGHT * 0.7)

        self.set_color(formulae_init)
        self.x_y_dist = (formulae_init[2].get_left()[0] - formulae_init[0].get_left()[0]) * 1.1

        self.position_xy(formulae_init)
        return formulae_init, formula_f

    def truncate_number(self, num, precision):
        self.truncate_number_count += 1
        scale = 10**precision
        res = round(num * scale) / scale
        print("trunc count: " + str(self.truncate_number_count) + ", before trunc: " + str(num) + str(", after: ") + str(res))
        return res

    def play_dynamic_stuff(self, updates_initial, formula_f, formulae_init=None):
        e_grad_tracker = ValueTracker(0)

        def get_step(e):
            e_val = e + 0.1
            return self.steps[math.floor(e_val)]

        if formulae_init is not None:
            self.play(FadeOut(formulae_init), run_time=0.01)

        def get_formulae():
            formulae = align_left(self.set_color(align_top(self.position_xy(MathTex(r'x &=',
                                                                                    str('{:2.2f}'.format(
                                                                                        self.truncate_number(get_step(
                                                                                            e_grad_tracker.get_value())[
                                                                                                                 0],
                                                                                                             2))),
                                                                                    r'\,\,\,\,' + r'y =',
                                                                                    str('{:2.2f}'.format(
                                                                                        self.truncate_number(get_step(
                                                                                            e_grad_tracker.get_value())[
                                                                                                                 1],
                                                                                                             2))) + r'\\',
                                                                                    r'\frac{\partial f}{\partial x} &= ' + str(
                                                                                        2 * self.c) + r'x',
                                                                                    r'=',
                                                                                    str('{:2.2f}'.format(
                                                                                        2 * self.c *
                                                                                        self.truncate_number(get_step(
                                                                                            e_grad_tracker.get_value())[
                                                                                                                 0],
                                                                                                             2))) + r'\\',
                                                                                    r'\frac{\partial f}{\partial y} &= ' + str(
                                                                                        2 * self.c) + r'y',
                                                                                    r'=',
                                                                                    str('{:2.2f}'.format(
                                                                                        2 * self.c *
                                                                                        self.truncate_number(get_step(
                                                                                            e_grad_tracker.get_value())[
                                                                                                                 1],
                                                                                                             2))),
                                                                                    font_size=self.font_size)
                                                                            )
                                                           .shift(RIGHT * self.horizontal_shift),
                                                           formula_f)
                                                 ),
                                  updates_initial
                                  )
            if self.f_0_position is not None:
                diff = self.f_0_position - formulae[0].get_center()
                formulae.shift(diff)
            else:
                self.f_0_position = formulae[0].get_center()
            return formulae

        formulae = always_redraw(lambda: get_formulae())
        self.add(formulae)

        # Add arrows.
        n_steps = self.n_steps_to_play
        steps = self.get_steps_for_plotting(n_steps)

        ttt = 15
        self.play(AnimationGroup(*[Create(step) for step in steps], lag_ratio=1),
                  e_grad_tracker.animate.set_value(n_steps), run_time=ttt, rate_func=linear)

    def get_z_label(self, font_size, color):
        fff = MathTex(r'f(x, y)', r' = 0.00', font_size=font_size, color=color)
        shift = np.array([-2.05, 1.45, 0]) - fff[0].get_center()
        fff.shift(shift)
        fff[0].set_color(color)
        fff[1].set_color(BLACK)
        return fff

    def construct(self):
        formulae_init, formula_f = self.setup_static_stuff()
        self.wait(4.2)
        self.play(Write(formulae_init[4]), Write(formulae_init[7]), run_time=2)
        interval_t = 0.1
        self.wait(4.5)
        self.play(Write(formulae_init[0]), Write(formulae_init[1]), Write(formulae_init[2]), Write(formulae_init[3]),
                  run_time=1.5)
        self.wait(5.5)
        self.play(Write(formulae_init[5]), Write(formulae_init[6]))
        self.wait(0.3)
        self.play(Write(formulae_init[8]), Write(formulae_init[9]))
        self.wait(5.5)

        updates_initial = self.get_updates().next_to(formulae_init, DOWN * 2)
        align_left(updates_initial, formulae_init)

        self.play(Write(updates_initial[0]))
        self.play(Write(updates_initial[1]))

        self.play_dynamic_stuff(updates_initial, formula_f, formulae_init=formulae_init)


class GradDescent(Execution):
    def __init__(self, other_scene=None, other_coords=None):
        super().__init__()
        self.ll = 7
        self.dist_formula_graph = 7.5
        self.plot_f_value = False
        self.f_val = None
        self.prev_f_vals = []
        self.init_point_colors = [BLUE, RED]
        self.init_point_current_color = 0
        self.e_arrow = ValueTracker(0)
        self.times_color_changed = 0
        self.other_scene = other_scene
        self.other_coords = other_coords
        self.x = -1
        self.y = -0.35
        if other_scene is not None:
            factor = 0.9
            self.x = self.x * factor
            self.y = self.y * factor

    def get_init_point(self):
        print("get_init_point 3")
        args = self.get_args(self.e_arrow.get_value())
        f_val = self.f(args[0], args[1])
        delta_ = 1e-4
        print("arg: " + str(self.e_arrow.get_value()))
        if len(self.prev_f_vals) >= 2 and abs(f_val - self.prev_f_vals[-1]) > delta_ \
                and (f_val - self.prev_f_vals[-1]) > delta_ \
                and (self.prev_f_vals[-1] - self.prev_f_vals[-2]) < - delta_\
                and self.e_arrow.get_value() < 0.05:
            print("1 cond satisfied")
            if self.times_color_changed == 1:
                self.init_point_current_color = 1 - self.init_point_current_color
                print("1 cond satisfied, init_point_current_color: " + str(self.init_point_current_color))
            self.times_color_changed += 1
        self.prev_f_vals.append(f_val)
        p = np.array([args[1], f_val, args[0]])
        col = 0 if self.e_arrow.get_value() < 0.5 else 1
        return Sphere(radius=self.r * 0.6).set_color(self.init_point_colors[col]).\
            move_to(self.axes.c2p(*p))

    def adjust_labels(self, label_x, label_y, label_z):
        label_x.shift(np.array([-0.75, 0, 0]))
        xx = 2.5
        yy = -3
        label_y.shift(np.array([xx, yy, 0]))
        label_x.shift(UP * 0.1)
        label_y.shift(UP * 0.4)
        label_y.shift(LEFT * 0.2)
        label_z.shift(UP * 1)

    def get_z_helper(self, str1, str2, color1, color2, font_size):
        fff = MathTex(str1, font_size=font_size)
        shift = np.array([-2.05, 1, 0]) - fff[0].get_center()
        fff.shift(shift)
        fff[0].set_color(color1)
        return fff

    def get_z_label(self, font_size, color):
        str1 = r'L(a, b)'
        args = self.get_args(self.e_arrow.get_value())
        f_val = self.f(args[0], args[1])
        str2 = r' = ' + str("{:2.2f}".format(f_val)) if self.plot_f_value else r' = 0.00'
        color2 = RED if self.plot_f_value else BLACK
        return self.get_z_helper(str1=str1, str2=str2, color1=color, color2=color2, font_size=font_size)

    def get_args(self, t):
        init_ = np.array([self.x, self.y])
        target_ = -init_
        point = t * target_ + (1 - t) * init_
        print("get_args: " + "t = " + str(t) + ", point = " + str(point))
        return point

    def construct(self):
        self.set_camera_orientation(zoom=0.8)
        min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
        point_r, point_color, point_removal_mode = get_params()

        min_x = 0
        max_x = 50
        a = 1000
        b = 30000
        point_r = point_r * 1.8

        def f(x):
            return a * x + b

        n_points = 12
        x_points = np.random.uniform(min_x, max_x, n_points)
        y_points = [f(x) for x in x_points]
        points = np.array([[x, y] for x, y in zip(x_points, y_points)]).T
        sigma_x = 6
        distorted_points = np.random.randn(*points.shape) * np.array([[sigma_x], [sigma_x * a]]) + points
        distorted_points = distorted_points[:, np.where(np.logical_and(distorted_points[0, :] > 0,
                                                                       distorted_points[1, :] > 20000))[0]]
        x_points = list(distorted_points[0, :])
        y_points = list(distorted_points[1, :])
        smallest_distance = 100000
        idx1 = -1
        idx2 = -1
        for ii in range(len(x_points)):
            for jj in range(ii):
                if abs(x_points[ii] - x_points[jj]) < smallest_distance:
                    smallest_distance = abs(x_points[ii] - x_points[jj])
                    if y_points[ii] < y_points[jj]:
                        idx1, idx2 = ii, jj
                    else:
                        idx1, idx2 = jj, ii
        x_points[idx2] += 10
        left_most = 0
        y_points[left_most] = y_points[left_most] - 12000
        greatest_y_idx = np.argmax(np.array(y_points))
        y_points[greatest_y_idx] -= 7000
        x_arr = np.array(x_points)
        y_arr = np.array(y_points)
        sorted_idx = np.argsort(x_arr)
        x_points = list(x_arr[sorted_idx])
        y_points = list(y_arr[sorted_idx])
        is_above_line = [y > f(x) for x, y in zip(x_points, y_points)]
        shift_val = 10000
        for i in range(len(y_points)):
            if is_above_line[i]:
                y_points[i] += shift_val
            else:
                y_points[i] -= shift_val
        ax = Axes(x_range=[0, 50 + 1, 10], y_range=[0, 80000, 20000],
                  x_length=8, y_length=5.7, tips=True, axis_config={"color": BLUE, "include_ticks": True}) \
            .add_coordinates().shift(UP * 0.8)
        label_font = 35
        coord_labels = ax.get_axis_labels(x_label=Text(r"experience (years)", font_size=label_font),
                                          y_label=Text(r"salary ($)", font_size=label_font))
        points = [Circle(radius=point_r, fill_color=point_color, color=point_color, fill_opacity=1, stroke_width=2)
                      .move_to(ax.c2p(x_, y_)) for x_, y_ in zip(x_points, y_points)]
        point_creation_time = 0.6
        time_to_create_all_points = 4
        start_times = np.random.random(len(points)) * (time_to_create_all_points - point_creation_time)
        anims = [Succession(Wait(run_time=st), FadeIn(p, run_time=point_creation_time),
                            Wait(run_time=time_to_create_all_points - point_creation_time - st))
                 for p, st in zip(points, start_times)]

        line = ax.plot(lambda x: f(x)).set_color(YELLOW)
        long_text_template = TexTemplate(
            tex_compiler="xelatex",
            output_format='.xdv',
        )
        long_text_template.add_to_preamble(r"\usepackage{siunitx}\usepackage{stix2}")
        MathTex.set_default(tex_template=long_text_template)
        Tex.set_default(tex_template=long_text_template)
        highlights = [Circle(radius=point_r * 1.8, stroke_width=2).set_color(GREEN).move_to(p.get_center())
                      for p in points]

        def get_arrow(p, x, above_line):
            start_ = p.get_center()
            end_ = ax.c2p(x, f(x))
            line = Line(start=start_, end=end_, color=BLUE, buff=0.08).add_tip(tip_length=0.2,
                                                                               tip_shape=ArrowTriangleFilledTip)
            shift_ = 0.06
            if above_line:
                line.shift(DOWN * shift_)
            else:
                line.shift(UP * shift_)
            return line

        arrows = [get_arrow(p, x, above) for p, x, above in zip(points, x_points, is_above_line)]
        highlight_starting_times = [i * 0.3 for i in range(len(highlights))]
        errors = [abs(f(x) - y) for x, y in zip(x_points, y_points)]
        formula_color = YELLOW

        def format_error(val):
            scale = 100
            return int(val) // scale * scale

        def get_predicted_value_position(arr, is_above):
            horizontal_shift = LEFT * 3
            vertical_pos = arr.end + UP * 0
            return vertical_pos + horizontal_shift

        error_labels = [MathTex(r'a x_' + str(i + 1) + r' + b', font_size=formula_font).set_color(formula_color).
                            move_to(get_predicted_value_position(arr, above)) for error, arr, above, i in
                        zip(errors, arrows, is_above_line, range(len(errors)))]
        for error_lab, arr, above, err in zip(error_labels, arrows, is_above_line, errors):
            desired_horizontal_distance = 0.15
            if above:
                shift_ = LEFT * desired_horizontal_distance - LEFT * (error_lab.get_right() - arr.get_center())[0]
                if err >= 10000:
                    shift_ = shift_ + 0.1 * LEFT
            else:
                shift_ = RIGHT * desired_horizontal_distance - RIGHT * (error_lab.get_left() - arr.get_center())[0]
            error_lab.shift(shift_)
        highlight_time = 0.3
        arrow_time = 0.3
        error_label_time = 0.3
        error_strings = [str(format_error(error)) for error in errors]
        formula_strings = []
        for idx, err in enumerate(error_strings):
            formula_strings.append(err)
            formula_strings.append("^2")
            if idx < len(error_strings) - 1:
                formula_strings.append("+")
        formula = MathTex(r'L(a, b) = ', r'(a x_1 + b - \text{salary}_1)^2', r'+ (a x_2 + b - \text{salary}_2)^2',
                          r'+ \dots + ', r'(a x_9 + b - \text{salary}_9)^2', font_size=formula_font + 10). \
            next_to(ax, DOWN * 4.2).set_color(YELLOW)
        formula_left = formula.get_left()
        first_error_coords = formula[1].get_center()
        error_moving_time = 0.8
        plus_writing_time = 0.3
        power_write_time = 0.6
        formula_shift_time = 0.7
        formula_new = MathTex(r'L(a, b) = ', r'\frac{1}{9}\Big[', r'(a x_1 + b - \text{salary}_1)^2',
                              r'+ (a x_2 + b - \text{salary}_2)^2', r'+ \dots + ', r'(a x_9 + b - \text{salary}_9)^2',
                              r'\Big]', font_size=formula_font + 10).set_color(YELLOW)
        formula_new.shift(formula_left - formula_new.get_left())
        formula_shift = formula_new[2].get_center() - first_error_coords
        formula_parts_to_shift = [formula[i] for i in [1, 2, 3, 4]]

        general_formula = MathTex(r'f(x) = a', r'x', r'+ b', font_size=formula_font + 10)
        general_formula[1].set_color(RED)
        general_formula.shift(formula_left - general_formula.get_left())
        # up_shift = 0.5
        up_shift = 0.9
        general_formula.shift(UP * up_shift)

        predicted_error_1 = MathTex(r'a x_' + str(1) + r' + b', font_size=formula_font + 10).set_color(YELLOW). \
            move_to(np.array([-3.1, 1, 0]))
        dd = -0.57
        true_error_1 = MathTex(r'\text{salary}_1', font_size=formula_font + 10).set_color(YELLOW). \
            move_to(np.array([-3.1, dd, 0]))
        d_shift = 0.25
        start_ = predicted_error_1.get_right() + LEFT * 0.05 + DOWN * d_shift
        l_shift = 1.15
        end_ = start_ + LEFT * l_shift
        predicted_error_line = Line(start=start_, end=end_)
        d_shift_2 = 1.05
        l_shift_2 = 1
        ll = 0.15
        start_ = start_ + DOWN * d_shift_2 + LEFT * ll
        end_ = start_ + LEFT * l_shift_2
        true_error_line = Line(start=start_, end=end_)

        ###############################################################################################################x
        self.play(Create(ax), Write(coord_labels), Write(formula[0]), Write(general_formula), run_time=1)
        self.play(*anims, run_time=1)
        self.play(Write(line), run_time=1.7)
        self.play(
            *[Succession(Wait(run_time=t), Write(highlight, run_time=highlight_time), Write(arrow, run_time=arrow_time))
              for highlight, arrow, t in zip(highlights, arrows, highlight_starting_times)])
        self.play(Write(predicted_error_line), run_time=1)
        self.play(Write(predicted_error_1), run_time=1)
        self.wait(0.1)
        self.play(Write(true_error_line), run_time=1)
        self.play(Write(true_error_1), run_time=1)
        formula_part_t = 1
        self.play(Write(formula[1]), run_time=formula_part_t)
        self.play(Write(formula[2]), run_time=formula_part_t)
        self.play(Write(formula[3]), run_time=formula_part_t)
        self.play(Write(formula[4]), run_time=formula_part_t)

        self.play(*[p.animate(run_time=formula_shift_time).shift(formula_shift) for p in formula_parts_to_shift])
        self.play(Write(formula_new[1]), Write(formula_new[-1]), run_time=1)
        self.wait(4)
        obj_for_destruction = [ax, coord_labels, line, predicted_error_line, predicted_error_1, true_error_line,
                               true_error_1] + highlights + arrows + points
        self.wait(1)
        self.play(*[Uncreate(obj) for obj in obj_for_destruction], run_time=1.5)
        up_dist = 7
        to_move = [formula, formula_new[1], formula_new[-1], general_formula]
        self.play(*[m.animate.shift(UP * up_dist) for m in to_move], run_time=1)
        self.wait(5.8)

        ######################################################################################################
        # StepSize

        f_formula = self.axes_graph_setup(other_scene=self.other_scene, other_coords=self.other_coords)
        init_point = always_redraw(lambda: self.get_init_point())
        if self.other_scene is None:
            self.play(Create(init_point))
        else:
            self.other_scene.play(Create(init_point))
        self.wait(1)
        target_step = self.steps[1]
        target_point = self.axes.c2p(*self.invert_x_y(np.array([target_step[0], self.f(target_step[0], target_step[1]),
                                                               target_step[1]])))

        def get_arg_dynamic(t):
            init_point_ = init_point if isinstance(init_point, np.ndarray) else init_point.get_center()
            target_point_ = target_point if isinstance(target_point, np.ndarray) else target_point.get_center()
            return t*target_point_ + (1-t)*init_point_

        def get_arrow_dynamic(t, add_pointer=True, buff=0.05):
            init_point_ = init_point if isinstance(init_point, np.ndarray) else init_point.get_center()
            target_point_ = target_point if isinstance(target_point, np.ndarray) else target_point.get_center()
            new_end = t*target_point_ + (1-t)*init_point_
            print("t: " + str(t))
            if t <= 0.5:
                return get_arrow_new(init_point_, new_end, add_pointer=add_pointer, color=BLUE, buff=buff)
            else:
                return get_arrow_new(init_point_, new_end, add_pointer=add_pointer, color=RED, buff=buff)

        def get_point_for_curve(t):
            point = self.get_args(t)
            self.f_val = self.f(point[0], point[1])
            self.plot_f_value = True
            return self.axes.c2p(*self.invert_x_y(np.array([point[0], self.f(point[0], point[1]), point[1]])))

        def get_curve_new(t, color, opacity=1):
            return ParametricFunction(get_point_for_curve, t_range=np.array([0, t]),
                               stroke_opacity=opacity, fill_opacity=0, stroke_width=4.5).set_color(color)

        def draw_full_curve(t):
            return get_curve_new(t, BLUE)

        def draw_half_curve(t):
            t = min(t, 0.5)
            return get_curve_new(t, BLUE)

        def draw_opt_point():
            res = 5
            r = 0.1
            opt_point = np.array([0, 0])
            coords = self.axes.c2p(*self.invert_x_y(np.array([opt_point[0], self.f(opt_point[0], opt_point[1]),
                                                             opt_point[1]])))
            return Sphere(
                center=coords,
                radius=r,
                resolution=(res, res),
                u_range=[0.001, PI - 0.001],
                v_range=[0, TAU]
            ).set_color(YELLOW).set_opacity(1)

        half_curve = always_redraw(lambda: draw_half_curve(self.e_arrow.get_value()))
        full_curve = always_redraw(lambda: draw_full_curve(self.e_arrow.get_value()))
        if self.other_scene is None:
            self.add(full_curve, half_curve)
        else:
            self.other_scene.add(full_curve, half_curve)

        ttt = 4
        if self.other_scene is None:
            self.play(self.e_arrow.animate.set_value(0.499), run_time=ttt, rate_func=linear)
        else:
            self.other_scene.play(self.e_arrow.animate.set_value(0.499), run_time=ttt, rate_func=linear)

        grads = MathTex(r'- \nabla L(a, b)', r' = - \Big[ \frac{\partial L(a, b)}{\partial a}, \frac{\partial L(a, b)}{\partial b} \Big]',
                        font_size=formula_font + 15).move_to([3, 0, 0]).set_color(YELLOW)
        grads.shift(2 * RIGHT)
        self.wait(2.3)
        self.play(Write(grads[0]), run_time=1.2)
        self.wait(2)
        self.play(Write(grads[1]), run_time=1.5)
        self.wait(8.3)


class Exec(LinearRegression):
    def __init__(self):
        super().__init__()
        self.e_vals = []
        self.small = True
        self.draw_function_value = False
        self.factor_ = 1000
        self.st = 1
        self.ll = 8
        self.axes_shift = 2.2
        self.axes_counter_shift = 1
        if self.small:
            self.x = -0.9
            self.y = 2
        else:
            self.x = -0.9
            self.y = 70
        self.last_displayed_a = -100000
        self.last_displayed_b = -100000
        self.r = 0.18
        self.c = 1.1
        self.dist_formula_graph = 7
        self.range = 1.2
        self.coord_range = 2
        self.height = 1.1
        self.horizontal_shift = 3
        self.font_size = 43
        self.step_size = 1
        self.formulas_color = YELLOW
        self.numbers_color = RED
        self.highlight_color = GREEN
        self.axes = None
        self.surf = None
        self.steps = None
        self.grads = None
        self.points = None
        self.x_y_dist = None
        self.truncate_number_count = 0
        self.f_0_position = None
        self.n_steps_to_play = 110
        self.f = None
        self.df_du = None
        self.df_dv = None


    def extra_rotation(self, m):
        return m


    def adjust_labels(self, label_x, label_y, label_z):
        pass

    def get_axes(self):
        ax = ThreeDAxes(x_range=[-self.coord_range, self.coord_range, self.st], x_length=self.ll,
                        y_range=[-self.coord_range * self.height, self.coord_range * self.height, self.st],
                        y_length=self.ll * self.height,
                        z_range=[-self.coord_range, self.coord_range, self.st], z_length=self.ll)
        if hasattr(self, 'graph_scale'):
            ax.scale(self.graph_scale)
        return ax

    def normalize_quadratic(self, u, v):
        vec = np.array([u, v])
        len = np.linalg.norm(vec)
        if len > self.range:
            vec = vec / len * self.range
        u = vec[0]
        v = vec[1]
        return u, v

    def get_surf(self, ax, res=None):
        opacity = 1
        if res is None:
            res = 20
        u_range = [-self.range, self.range]
        v_range = [-self.range, self.range]
        return Surface(
            lambda u, v: ax.
                c2p(self.normalize_quadratic(u, v)[0], self.f(self.normalize_quadratic(u, v)[0],
                                                              self.normalize_quadratic(u, v)[1]),
                    self.normalize_quadratic(u, v)[1]),
            u_range=u_range, v_range=v_range, checkerboard_colors=[GREEN_D, GREEN_E],
            resolution=(res, res)).set_opacity(opacity)

    def grad_descent(self, x, y):
        n_iters = 300 if self.small else 12000
        step_size = self.step_size
        step_size = 0.1 if self.small else 0.0001
        self.steps = []
        self.grads = []
        for iter in range(n_iters):
            dx = self.df_du(x, y)
            dy = self.df_dv(x, y)
            self.steps.append([x, y])
            grad = np.array([dx, dy])
            self.grads.append(grad)
            # Line search
            if not self.small:
                step_size = 1
                beta = 0.5
                val = self.f(x - step_size * grad[0], y - step_size * grad[1])
                step_size = beta * step_size
                val_new = self.f(x - step_size * grad[0], y - step_size * grad[1])
                while val_new < val:
                    step_size = beta * step_size
                    val = val_new
                    val_new = self.f(x - step_size * grad[0], y - step_size * grad[1])
                step_size = step_size / beta
            x = x - step_size * grad[0]
            y = y - step_size * grad[1]
            print("f: " + str(self.f(x, y)) + ", a: " + str(x) + ", b: " + str(y))
        self.steps.append([x, y])

    def xy_to_3d(self, two_d_vec):
        x = two_d_vec[0]
        y = two_d_vec[1]
        return x, self.f(x, y), y

    def axes_graph_setup(self, angle_y_axis=0, other_scene=None, other_coords=None):
        axis_label_color = BLUE
        lab_font_size = 46
        formula_f = MathTex(r'f(x, y) &= ' + str(self.c) + r'(x^2 + y^2)', font_size=self.font_size) \
            .set_color(self.formulas_color).set_opacity(0).\
            move_to(np.array([3, 0, 0])).shift(self.axes_counter_shift * DOWN).shift(LEFT)
        self.play(Write(formula_f), run_time=0.001)
        return formula_f

    def get_init_point_helper(self):
        i = 0
        vec = self.steps[i]
        u = vec[0]
        v = vec[1]
        return self.axes.c2p(*self.invert_x_y(np.array([u, self.f(u, v), v])))

    def get_init_point(self):
        prev_point = self.get_init_point_helper()
        return Sphere(radius=self.r).set_color(YELLOW).move_to(prev_point)

    def get_updates(self):
        return MathTex(r'a &:= a - \beta \frac{\partial L}{\partial a}\\',
                       r'b &:= b - \beta \frac{\partial L}{\partial b}',
                       font_size=self.font_size).set_color(self.formulas_color)

    def invert_x_y(self, vec):
        return np.array([vec[2], vec[1], vec[0]])

    def get_steps_for_plotting(self, n_steps):
        steps = VGroup()
        for idx in range(n_steps):
            prev_point = self.steps[idx]
            currect_point = self.steps[idx + 1]
            color = RED if (np.array(currect_point) - np.array(prev_point))[0] > 0 else BLUE
            new_step = Line(start=self.axes.c2p(*self.invert_x_y(self.xy_to_3d(prev_point))),
                            end=self.axes.c2p(*self.invert_x_y(self.xy_to_3d(currect_point))), color=color, buff=0.05). \
                add_tip(tip_shape=ArrowTriangleFilledTip, tip_length=0.2)
            steps.add(new_step)
        return steps

    def set_color(self, m):
        m[0].set_color(self.formulas_color)
        m[1].set_color(self.numbers_color)
        m[2].set_color(self.formulas_color)
        m[3].set_color(self.numbers_color)
        m[4].set_color(self.formulas_color)
        m[5].set_color(self.formulas_color)
        m[6].set_color(self.numbers_color)
        m[7].set_color(self.formulas_color)
        m[8].set_color(self.formulas_color)
        m[9].set_color(self.numbers_color)
        return m

    def position_xy(self, m):
        current_dist = m[2].get_left()[0] - m[0].get_left()[0]
        coords = m[2].get_center()
        m[2].move_to(np.array([coords[0] + self.x_y_dist - current_dist, coords[1], coords[2]]))

        dist = m[1].get_left()[0] - m[0].get_right()[0]
        dist_current = m[3].get_left()[0] - m[2].get_right()[0]
        m[3].shift(RIGHT * (dist - dist_current))
        return m

    def setup_static_stuff(self):
        formula_f = self.axes_graph_setup()

        formulae_init = MathTex(r'a &=',
                                str('{:2.2f}'.format(self.x)),
                                r'\,\,\,\,' + r'b =',
                                str('{:2.2f}'.format(self.y)) + r'\\',
                                r'\frac{\partial L}{\partial a} &= \frac{1}{9}\sum_{i=1}^9 2 x_i (x_i a + b - y_i)',
                                r'=',
                                str('{:4.2f}'.format(self.df_du(self.x, self.y))) + r'\\',
                                r'\frac{\partial L}{\partial b} &= \frac{1}{9}\sum_{i=1}^9 2(b + x_i a - y_i)',
                                r'=',
                                str('{:4.2f}'.format(self.df_dv(self.x, self.y))),
                                font_size=self.font_size).shift(RIGHT * self.horizontal_shift)
        align_top(formulae_init, formula_f)
        formulae_init.shift(RIGHT * 0.7).shift(LEFT * 0.2)

        self.set_color(formulae_init)
        self.x_y_dist = (formulae_init[2].get_left()[0] - formulae_init[0].get_left()[0]) * 1.1

        self.position_xy(formulae_init)
        v_shift = 5
        h_shift = 2.5
        formulae_init.shift(UP * v_shift + RIGHT * h_shift)
        formula_f.shift(UP * v_shift + RIGHT * h_shift)
        return formulae_init, formula_f

    def truncate_number(self, num, precision):
        self.truncate_number_count += 1
        scale = 10 ** precision
        res = round(num * scale) / scale
        print(
            "trunc count: " + str(self.truncate_number_count) + ", before trunc: " + str(num) + str(", after: ") + str(res))
        return res

    def generate_steps(self):
        step_prev = [-100000, -100000]
        for e in np.linspace(0, len(self.steps), num=10000):
            e_val = e + 0.001
            ee = math.floor(e_val)
            if ee <= len(self.steps) - 1:
                step = self.steps[ee]
                if np.linalg.norm(np.array(step) - np.array(step_prev)) > 0.00001:
                    self.e_vals.append(e)
                    step_prev = step

    def get_step(self, e, for_line=False):
        e_val = e
        step = self.steps[math.floor(e_val)]
        if for_line:
            idx = math.floor(e_val)
            idx_next = idx + 1
            if idx_next >= len(self.steps):
                return self.steps[-1]
            else:
                prev_step = self.steps[idx]
                next_step = self.steps[idx + 1]
                alpha = e_val - idx
                return list(alpha * np.array(next_step) + (1 - alpha) * np.array(prev_step))
        else:
            a_new = step[0]
            b_new = step[1]
            th = 0.03
            if abs(a_new - self.last_displayed_a) > th or abs(b_new - self.last_displayed_b) > th:
                self.last_displayed_a = a_new
                self.last_displayed_b = b_new
                return [a_new, b_new]
            else:
                return [self.last_displayed_a, self.last_displayed_b]

    def play_dynamic_stuff(self, updates_initial, formula_f, e_grad_tracker, formulae_init=None):

        if formulae_init is not None:
            self.play(FadeOut(formulae_init), run_time=0.01)

        def get_formulae():
            def get_a(e):
                return self.get_step(e_grad_tracker.get_value())[0]

            def get_b(e):
                return self.get_step(e_grad_tracker.get_value())[1]

            formulae = align_left(self.set_color(align_top(self.position_xy(MathTex(r'a &=',
                                                                                    str('{:2.2f}'.format(
                                                                                        self.truncate_number(self.get_step(
                                                                                            e_grad_tracker.get_value())[
                                                                                                                 0],
                                                                                                             2))),
                                                                                    r'\,\,\,\,' + r'b =',
                                                                                    str('{:2.2f}'.format(
                                                                                        self.truncate_number(self.get_step(
                                                                                            e_grad_tracker.get_value())[
                                                                                                                 1],
                                                                                                             2))) + r'\\',
                                                                                    r'\frac{\partial L}{\partial a} &= \frac{1}{9}\sum_{i=1}^9 2 x_i (x_i a + b - y_i)',
                                                                                    r'=',
                                                                                    str('{:4.2f}'.format(self.df_du(get_a(e_grad_tracker), get_b(e_grad_tracker)))) + r'\\',
                                                                                    r'\frac{\partial L}{\partial b} &= \frac{1}{9}\sum_{i=1}^9 2(b + x_i a - y_i)',
                                                                                    r'=',
                                                                                    str('{:4.2f}'.format(self.df_dv(get_a(e_grad_tracker), get_b(e_grad_tracker)))),
                                                                                    font_size=self.font_size)
                                                                            )
                                                           .shift(RIGHT * self.horizontal_shift),
                                                           formula_f)
                                                 ),
                                  updates_initial
                                  )
            if self.f_0_position is not None:
                diff = self.f_0_position - formulae[0].get_center()
                formulae.shift(diff)
            else:
                self.f_0_position = formulae[0].get_center()
            return formulae

        formulae = always_redraw(lambda: get_formulae())
        self.add(formulae)

        # Add arrows.
        n_steps = 300 if self.small else len(steps___)

        @unit_interval
        def rate_accelerating(t: float) -> float:
            return t ** 4

        self.play(e_grad_tracker.animate.set_value(n_steps), run_time=6.7, rate_func=rate_accelerating)

    def get_z_label(self, font_size, color):
        fff = MathTex(r'f(x, y)', r' = 0.00', font_size=font_size, color=color)
        shift = np.array([-2.05, 1.45, 0]) - fff[0].get_center()
        fff.shift(shift)
        fff[0].set_color(color)
        fff[1].set_color(BLACK)
        return fff

    def construct(self):
        e_grad_tracker = ValueTracker(0)
        a = self.x
        b = self.y

        self.camera.frame.set_width(self.camera.frame.get_width() * 1.4)
        min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
        point_r, point_color, point_removal_mode = get_params()

        min_x = 0
        max_x = 50
        a_ = 1000
        b_ = 30000
        point_r = point_r * 1.8

        def f(x):
            return a_ * x + b_

        n_points = 12
        x_points = np.random.uniform(min_x, max_x, n_points)
        y_points = [f(x) for x in x_points]
        points = np.array([[x, y] for x, y in zip(x_points, y_points)]).T
        sigma_x = 6
        distorted_points = np.random.randn(*points.shape) * np.array([[sigma_x], [sigma_x * a_]]) + points
        distorted_points = distorted_points[:, np.where(np.logical_and(distorted_points[0, :] > 0,
                                                                       distorted_points[1, :] > 20000))[0]]
        x_points = list(distorted_points[0, :])
        y_points = list(distorted_points[1, :])
        smallest_distance = 100000
        idx1 = -1
        idx2 = -1
        for ii in range(len(x_points)):
            for jj in range(ii):
                if abs(x_points[ii] - x_points[jj]) < smallest_distance:
                    smallest_distance = abs(x_points[ii] - x_points[jj])
                    if y_points[ii] < y_points[jj]:
                        idx1, idx2 = ii, jj
                    else:
                        idx1, idx2 = jj, ii
        x_points[idx2] += 10
        left_most = 0
        y_points[left_most] = y_points[left_most] - 12000
        greatest_y_idx = np.argmax(np.array(y_points))
        y_points[greatest_y_idx] -= 7000
        x_arr = np.array(x_points)
        y_arr = np.array(y_points)
        sorted_idx = np.argsort(x_arr)
        x_points = list(x_arr[sorted_idx])
        y_points = list(y_arr[sorted_idx])
        is_above_line = [y > f(x) for x, y in zip(x_points, y_points)]
        shift_val = 10000
        for i in range(len(y_points)):
            if is_above_line[i]:
                y_points[i] += shift_val
            else:
                y_points[i] -= shift_val

        if self.small:
            x_max_ = np.max(np.array(x_points))
            for i in range(len(x_points)):
                x_points[i] = x_points[i] / x_max_ * 1.7
            y_max_ = np.max(np.array(y_points))
            for i in range(len(y_points)):
                y_points[i] = y_points[i] / y_max_ * 2

        def get_f():
            def my_f(a, b):
                val = 0
                for x_, y_ in zip(x_points, y_points):
                    x = x_
                    y = y_ if self.small else y_ / 1000
                    temp1 = (x**2 * a**2 + b**2 + 2 * x * a * b - 2 * x * y * a - 2 * y * b + y**2)
                    temp2 = x**2 * a**2 + b**2 + 2*x*a*b - 2*x*y*a - 2*y*b + y**2
                    assert abs(temp1 - temp2) < 0.000001
                    val += temp1
                return val / 9
            return my_f

        def get_df_da():
            def dfa(a, b):
                val = 0
                for x_, y_ in zip(x_points, y_points):
                    x = x_
                    y = y_ if self.small else y_ / 1000
                    temp1 = 2 * x * (x * a + b - y)
                    temp2 = 2*x**2*a + 2*x*b - 2*x*y
                    assert abs(temp1 - temp2) < 0.000001
                    val += temp1
                return val / 9
            return dfa

        def get_df_db():
            def dfb(a, b):
                val = 0
                for x_, y_ in zip(x_points, y_points):
                    x = x_
                    y = y_ if self.small else y_ / 1000
                    temp1 = 2 * (b + x * a - y)
                    temp2 = 2*b + 2*x*a - 2*y
                    assert abs(temp1 - temp2) < 0.000001
                    val += temp1
                return val / 9
            return dfb

        self.f = get_f()
        self.df_du = get_df_da()
        self.df_dv = get_df_db()

        self.grad_descent(a, b)

        x_range = [0, 2, 0.5] if self.small else [0, 50 + 1, 10]
        y_range = [0, 2, 0.5] if self.small else [0, 80000, 20000]
        ax = Axes(x_range=x_range, y_range=y_range,
                  x_length=8, y_length=5.7, tips=True, axis_config={"color": BLUE, "include_ticks": True}) \
            .add_coordinates()
        ax.shift(LEFT * 3.5)
        new_ticks = []
        if not self.small:
            for obj in ax.y_axis.numbers:
                obj.set_opacity(0)
                sh_ = 0.3
                new_ticks.append(MathTex(r"{:.0f}\,\text{{k}}".format(obj.get_value() / 1000)).scale(0.8).
                         move_to(obj.get_right()).shift(LEFT * sh_))

        label_font = 35
        x_label__ = r'x' if self.small else r"experience (years)"
        y_label__ = r'y' if self.small else r"salary ($)"
        coord_labels = ax.get_axis_labels(x_label=Text(x_label__, font_size=label_font),
                                          y_label=Text(y_label__, font_size=label_font))
        x_lab_ = coord_labels[0]
        y_lab_ = coord_labels[1]
        y_lab_.shift(LEFT * 0.3)
        points = [Circle(radius=point_r, fill_color=point_color, color=point_color, fill_opacity=1, stroke_width=2)
                      .move_to(ax.c2p(x_, y_)) for x_, y_ in zip(x_points, y_points)]
        point_creation_time = 0.6
        time_to_create_all_points = 2
        start_times = np.random.random(len(points)) * (time_to_create_all_points - point_creation_time)
        fact = 1 if self.small else self.factor_
        line = always_redraw(lambda: ax.plot(lambda xx: fact * (self.get_step(e_grad_tracker.get_value(), for_line=True)[0] * xx +
                                                      self.get_step(e_grad_tracker.get_value(), for_line=True)[1])).set_color(YELLOW))
        long_text_template = TexTemplate(
            tex_compiler="xelatex",
            output_format='.xdv',
        )
        long_text_template.add_to_preamble(r"\usepackage{siunitx}\usepackage{stix2}")
        MathTex.set_default(tex_template=long_text_template)
        Tex.set_default(tex_template=long_text_template)
        idx = np.array([p.get_center()[0] for p in points]).argsort().argsort()

        shifts = [1 if a else -1 for a in is_above_line]
        data_labels = [MathTex(r'x_' + str(idx[i] + 1) + r', y_' + str(idx[i] + 1), font_size=formula_font+10).
                           move_to(p.get_center() + UP * shift * 0.3)
                       for (i, p), shift in zip(enumerate(points), shifts)]

        anims = [Succession(Wait(run_time=st), FadeIn(p, run_time=point_creation_time),
                            Write(data_label, run_time=0.3))
                 for p, st, data_label in zip(points, start_times, data_labels)]

        def get_arrow(p, x, above_line):
            start_ = p.get_center()
            end_ = ax.c2p(x, f(x))
            line = Line(start=start_, end=end_, color=BLUE, buff=0.08).add_tip(tip_length=0.2,
                                                                               tip_shape=ArrowTriangleFilledTip)
            shift_ = 0.06
            if above_line:
                line.shift(DOWN * shift_)
            else:
                line.shift(UP * shift_)
            return line

        data_lab_starting_times = [i * 0.3 for i in range(len(data_labels))]
        errors = [abs(f(x) - y) for x, y in zip(x_points, y_points)]
        formula_color = YELLOW

        def format_error(val):
            scale = 100
            return int(val) // scale * scale

        def get_predicted_value_position(arr, is_above):
            horizontal_shift = LEFT * 3
            vertical_pos = arr.end + UP * 0
            return vertical_pos + horizontal_shift

        data_label_time = 0.3
        arrow_time = 0.3
        error_strings = [str(format_error(error)) for error in errors]
        formula_strings = []
        for idx, err in enumerate(error_strings):
            formula_strings.append(err)
            formula_strings.append("^2")
            if idx < len(error_strings) - 1:
                formula_strings.append("+")
        r_shift = 1
        formula = MathTex(r'L(a, b) = ', r'(a x_1 + b - y_1)^2', r'+ (a x_2 + b - y_2)^2',
                          r'+ \dots + ', r'(a x_9 + b - y_9)^2', font_size=formula_font + 10). \
            next_to(ax, DOWN * 4.2).set_color(YELLOW).shift(RIGHT * r_shift)
        formula_left = formula.get_left()
        first_error_coords = formula[1].get_center()
        formula_shift_time = 0.7
        general_formula = MathTex(r'f(x) = a', r'x', r'+ b', font_size=formula_font + 10)
        general_formula[1].set_color(RED)
        general_formula.shift(formula_left - general_formula.get_left())
        up_shift = 0.9
        general_formula.shift(UP * up_shift)

        def get_main_formula():

            a = self.get_step(e_grad_tracker.get_value())[0]
            b = self.get_step(e_grad_tracker.get_value())[1]
            f_val = self.f(a, b)
            formula_part = '{:2.2f}'.format(self.truncate_number(f_val, 2))
            formula_new = MathTex(r'L(a, b) = ', r'\frac{1}{9}\Big[', r'(a x_1 + b - y_1)^2',
                                  r'+ (a x_2 + b - y_2)^2', r'+ \dots + ', r'(a x_9 + b - y_9)^2',
                                  r'\Big]', '=', str(formula_part), font_size=formula_font + 10).\
                set_color(YELLOW)
            formula_new.shift(formula_left - formula_new.get_left())
            formula_new[-1].set_color(RED)
            if self.draw_function_value:
                formula_new[-1].set_opacity(1)
                formula_new[-2].set_opacity(1)
            else:
                formula_new[-1].set_opacity(0)
                formula_new[-2].set_opacity(0)
            return formula_new

        formula_loss = always_redraw(lambda: get_main_formula())

        predicted_error_1 = MathTex(r'a x_' + str(1) + r' + b', font_size=formula_font + 10).set_color(YELLOW). \
            move_to(np.array([-3.1, 1, 0]))
        dd = -0.57
        true_error_1 = MathTex(r'\text{salary}_1', font_size=formula_font + 10).set_color(YELLOW). \
            move_to(np.array([-3.1, dd, 0]))
        d_shift = 0.25
        start_ = predicted_error_1.get_right() + LEFT * 0.05 + DOWN * d_shift
        l_shift = 1.15
        end_ = start_ + LEFT * l_shift
        predicted_error_line = Line(start=start_, end=end_)

        d_shift_2 = 1.05
        l_shift_2 = 1
        ll = 0.15
        start_ = start_ + DOWN * d_shift_2 + LEFT * ll
        end_ = start_ + LEFT * l_shift_2
        true_error_line = Line(start=start_, end=end_)

        ###############################################################################################################x
        self.play(Create(ax), *[Write(t) for t in new_ticks], Write(coord_labels), Write(formula_loss), Write(general_formula), run_time=1)
        self.play(*anims, run_time=1)

        ################################################################################################

        formulae_init, formula_f = self.setup_static_stuff()
        self.wait(2.5)
        self.play(Write(formulae_init[4]), Write(formulae_init[7]), run_time=1.5) #7s
        interval_t = 0.1
        self.wait(1.5)
        self.play(Write(formulae_init[0]), Write(formulae_init[1]), Write(formulae_init[2]), Write(formulae_init[3]),
                  run_time=1.5)
        self.draw_function_value = True
        self.play(Write(line), run_time=1.5)
        self.play(Write(formulae_init[5]), Write(formulae_init[6]))
        self.wait(0.3)
        self.play(Write(formulae_init[8]), Write(formulae_init[9]))

        updates_initial = self.get_updates().next_to(formulae_init, DOWN * 2)
        align_left(updates_initial, formulae_init)

        self.play(Write(updates_initial[0]), run_time=1)
        self.play(Write(updates_initial[1]), run_time=1)

        self.wait(2)

        self.play_dynamic_stuff(updates_initial, formula_f, e_grad_tracker, formulae_init=formulae_init)


class Thumbnail(MovingCameraScene):
    def construct(self):
        min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn, init_a, init_b, disturb_coeff, \
        point_r, point_color, point_removal_mode = get_params()

        n_points_at_once = math.floor(n_points_at_once * 1.5)
        disturb_coeff *= 2

        init_b = 1.2
        n_updates = 3

        lr_simulator = LinearRegressionSimulator(min_x, max_x, n_points_at_once, n_updates, n_points_to_update_per_turn,
                                                 init_a, init_b, disturb_coeff, point_removal_mode)
        a, b, x_points, y_points, added, removed = lr_simulator.get_data()
        min_x_lim, max_x_lim, min_y_lim, max_y_lim = lr_simulator.get_limits()

        ax = Axes(x_range=[min_x_lim - 1, max_x_lim + 1, 1], y_range=[min_y_lim - 1, max_y_lim + 1, 1],
                  x_length=12, y_length=8, tips=True, axis_config={"color": BLUE, "include_ticks": False}) \
            .shift(DOWN * 0.5)
        points = [Circle(radius=point_r, fill_color=point_color, color=point_color, fill_opacity=1, stroke_width=2)
                      .move_to(ax.c2p(x_, y_)) for x_, y_ in zip(x_points, y_points)]
        point_creation_time = 1.6
        time_between_rounds = 0.9
        anims = []
        a = np.array([lr_simulator.a[0]])
        b = np.array([lr_simulator.b[0]])
        range_length = 100
        for i in range(1, len(lr_simulator.a)):
            new_range_a = np.linspace(lr_simulator.a[i-1], lr_simulator.a[i], range_length)
            a = np.hstack([a, new_range_a[1:]])
            new_range_b = np.linspace(lr_simulator.b[i-1], lr_simulator.b[i], range_length)
            b = np.hstack([b, new_range_b[1:]])
        line_progress = ValueTracker(0)

        def get_index(e):
            return math.floor((len(a) - 1) * e)

        line = always_redraw(lambda: ax.plot(lambda x: a[get_index(line_progress.get_value())] * x +
                                                       b[get_index(line_progress.get_value())]).set_color(YELLOW))
        total_time = 0
        creation_anim_starting_times = []
        destruction_anim_starting_times = []
        for t in range(n_updates + 1):
            for _ in added[t]:
                creation_anim_starting_times.append(total_time)
            for _ in removed[t]:
                destruction_anim_starting_times.append(total_time)
            total_time = total_time + time_between_rounds
        for idx in range(len(creation_anim_starting_times)):
            start_create = creation_anim_starting_times[idx]
            end_create = start_create + point_creation_time
            create_anim = FadeIn(points[idx], run_time=end_create,
                                 rate_func=delayed_rate(delay=start_create/end_create))
            if idx < len(destruction_anim_starting_times):
                start_destroy = destruction_anim_starting_times[idx]
                end_destroy = start_destroy + point_creation_time
                destroy_anim = FadeOut(points[idx], run_time=end_destroy - start_destroy)
                color_change_1 = points[idx].animate(run_time=(start_destroy - end_create) / 2).set_color([YELLOW])
                color_change_2 = points[idx].animate(run_time=(start_destroy - end_create) / 2).set_color([PURPLE])
                anims.append(Succession(create_anim, color_change_1, color_change_2, destroy_anim))
            else:
                anims.append(create_anim)
        self.add(ax, line)
        t = 2
        line_animation = Succession(Transform(line_progress, line_progress, run_time=t),
                                    line_progress.animate(run_time=n_updates * time_between_rounds - t,
                                               rate_func=linear).set_value((n_updates * time_between_rounds - t) /
                                                                           n_updates / time_between_rounds))
        self.play(*anims, line_animation)
        self.wait(0.1)