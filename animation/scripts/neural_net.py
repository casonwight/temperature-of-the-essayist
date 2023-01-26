from manim import *
import numpy as np
import os

ASSET_PATH = "animation/assets/"


class FunctionEstimator(Scene):
    def ticker_anim(self, inputs, outputs, func, pause_time=.5):

        input_width = max([i.get_width() for i in inputs])
        output_width = max([o.get_width() for o in outputs])

        left = func[0].copy()
        mid = func[1].copy()

        first_group = (
            VGroup(left, mid)
            .arrange(RIGHT, buff=input_width+1)
            .move_to(ORIGIN)
            .shift(output_width/2 * LEFT)
        )
        
        for input, output in zip(inputs, outputs):
            input.move_to(first_group.get_center())
            output.next_to(mid, RIGHT, buff=1/2)        


        self.play(
            Transform(func[0], left),
            Transform(func[1], mid)
        )
        self.wait(pause_time)

        self.play(FadeIn(inputs[0], shift=UP), FadeIn(outputs[0], shift=UP))
        self.wait(pause_time)

        for old_x, old_y, x, y in zip(inputs[:-1], outputs[:-1], inputs[1:], outputs[1:]):
            self.play(
                FadeOut(old_x, shift=UP),
                FadeOut(old_y, shift=UP),
                FadeIn(x, shift=UP),
                FadeIn(y, shift=UP)
            )
            self.wait(pause_time)   
        
        self.play(
            FadeOut(inputs[-1], shift=UP),
            FadeOut(outputs[-1], shift=UP)
        )

        return func

    def construct(self):
        self.camera.background_color = WHITE

        func_wrap = VGroup(
            Tex("$f($", color=BLACK),
            Tex("$)=$", color=BLACK)
        ).arrange(RIGHT, buff=1).move_to(ORIGIN)

        # Pictures
        cat1 = ImageMobject(ASSET_PATH + "cat-1.jpg").scale_to_fit_width(5)
        cat2 = ImageMobject(ASSET_PATH + "cat-2.jpg").scale_to_fit_width(5)
        dog1 = ImageMobject(ASSET_PATH + "dog-1.jpg").scale_to_fit_width(5)
        animals = [cat1, dog1, cat2]
        labs_vals = [r"\text{cat}", r"\text{dog}", r"\text{cat}"]
        labels = VGroup(*[Tex(lab, color=RED) for lab in labs_vals])

        func_wrap = self.ticker_anim(animals, labels, func_wrap, pause_time=3/4)

        # Chess
        best_move_vals = [r"\text{pawn takes e5}", r"\text{rook a to d8}", r"\text{pawn to h4}"]
        boards = [ImageMobject(ASSET_PATH + "board-" + str(i) + ".jpeg").scale_to_fit_width(5) for i in range(1, 4)]
        best_moves = [Tex(lab, color=RED) for lab in best_move_vals]

        func_wrap = self.ticker_anim(boards, best_moves, func_wrap, pause_time=1)

        # Text
        prompts = [
            '``To be or not "',
            '``It was the best of times, "',
            '``I am no bird, and "'
        ]
        responses = [
            '``to be"',
            '``it was the worst of times"',
            '``no net ensnares me"'
        ]
        prompts = [Tex(p, color=BLACK).scale(.75) for p in prompts]
        responses = [Tex(r, color=RED).scale(.75) for r in responses]

        func_wrap = self.ticker_anim(prompts, responses, func_wrap, pause_time=1)
        
        func_wrap2 = func_wrap.copy().arrange(RIGHT, buff=1).move_to(ORIGIN)
        self.play(Transform(func_wrap, func_wrap2))