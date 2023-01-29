from manim import *
from scipy.special import softmax
import numpy as np

class Temperature(Scene):
    @staticmethod
    def raw_to_prob(raw_vals, temp=1.0):
        return list(np.round(softmax(np.array(raw_vals) / temp) * 100, 0).astype(int))

    def construct(self):
        self.camera.background_color = WHITE

        temp = 1.0

        og_values = [.1, -.3, .5, .2, -.1]

        chart = BarChart(
            values=self.raw_to_prob(og_values, temp),
            bar_names=["word 1", "word 2", "word 3", "word 4", "word 5"],
            y_range=[0, 100, 20],
            y_axis_config={"color": BLACK},
            x_axis_config={"include_ticks": False, "color": BLACK},
            y_length=6,
            x_length=10,
            color=BLACK
        ).scale(5/6)
        title = Text("Effect of Temperature", color=BLACK).next_to(chart, UP, buff=1/4)
        y_label = chart.get_y_axis_label("Probability").set_color(BLACK).scale(3/4).rotate(PI / 2).next_to(chart, LEFT)
        
        chart.x_axis.set_color(BLACK)
        chart.y_axis.set_color(BLACK)
        for i, num in enumerate(chart.y_axis.numbers):
            chart.y_axis.numbers[i] = Tex(f"{num.get_value():.0f}\%", color=BLACK).scale(1/2).move_to(num.get_center())

        
        temp_title = Tex("Temperature", color=BLACK).scale(1/2).next_to(chart, RIGHT).align_to(chart, UP)
        temp_display = DecimalNumber(1.0, num_decimal_places=2).next_to(temp_title, DOWN).set_color(BLACK)
        self.add(chart, y_label, title, temp_title, temp_display)        
        self.wait(2)
        temp_tracker = ValueTracker(0.0)   
        temp_display.add_updater(lambda d: d.set_value(temp_tracker.get_value()))
        temp_tracker.set_value(float(temp))  

        

        for temp in [.2, 8.50, .01, 1.25, 100.0, 1.0]:
            self.play(
                chart.animate.change_bar_values(self.raw_to_prob(og_values, temp)),
                temp_tracker.animate.set_value(float(temp)) 
            )
            self.wait(2)

        