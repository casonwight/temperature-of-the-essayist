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
        """ The audience doesn't know what a neural network is, so we need to start from the beginning.
        We want to show that a neural network is a universal function approximator, and that it can be used
        to approximate any function. We want to show this by showing that it can approximate a function that
        takes in a picture, a chess board, and a text prompt, and outputs a response.
        """
        self.camera.background_color = WHITE

        # Pictures
        cat1 = ImageMobject(ASSET_PATH + "cat-1.jpg").scale_to_fit_width(5)
        cat2 = ImageMobject(ASSET_PATH + "cat-2.jpg").scale_to_fit_width(5)
        dog1 = ImageMobject(ASSET_PATH + "dog-1.jpg").scale_to_fit_width(5)
        animals = [cat1, dog1, cat2]
        labs_vals = [r"\text{cat}", r"\text{dog}", r"\text{cat}"]
        labels = VGroup(*[Tex(lab, color=RED) for lab in labs_vals])

        func_wrap = VGroup(
            Tex("$f($", color=BLACK),
            Tex("$)=$", color=BLACK)
        ).arrange(RIGHT, buff=5 + 1).move_to(ORIGIN).shift(labels[1].get_width()/2 * LEFT)
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
        
        func_wrap2 = func_wrap.copy().arrange(RIGHT, buff=5 + 1).move_to(ORIGIN).shift(labels[1].get_width()/2 * LEFT)
        self.play(Transform(func_wrap, func_wrap2))

class SimplifiedNeuralNetwork(Scene):
    def construct(self):
        """ After showing that neural networks are universal function approximators, 
        we show how a simple neural network is structured.
        We also want to show that as a network sees example input-output pairs, it 
        makes adjustments to its weights and biases to better approximate the function.
        
        We want to use word prediction as the example, because we will move to transformers
        Everything should be pretty simple, because this is a non-technical audience.
        """
        self.camera.background_color = WHITE

        # Show a network with 5 inputs, 2 hidden layers, and 5 output
        inputs = VGroup(*[Circle(color=BLACK).set_fill(BLACK, opacity=.5) for _ in range(3)]).arrange(DOWN)
        hidden1 = VGroup(*[Circle(color=BLUE).set_fill(BLUE, opacity=.5) for _ in range(5)]).arrange(DOWN)
        hidden2 = VGroup(*[Circle(color=BLUE).set_fill(BLUE, opacity=.5) for _ in range(5)]).arrange(DOWN)
        outputs = VGroup(*[Circle(color=RED).set_fill(RED, opacity=.5) for _ in range(3)]).arrange(DOWN)
        neurons = VGroup(inputs, hidden1, hidden2, outputs).arrange(RIGHT, buff=2)
        input_label = Tex("Inputs", color=BLACK).next_to(inputs, UP)
        output_label = Tex("Outputs", color=BLACK).next_to(outputs, UP)
        neurons.add(input_label, output_label)

        input_hidden1_lines = VGroup(*[Line(input.get_right(), output.get_left(), color=GREY_A) for input in inputs for output in hidden1])
        hidden1_hidden2_lines = VGroup(*[Line(input.get_right(), output.get_left(), color=GREY_A) for input in hidden1 for output in hidden2])
        hidden2_output_lines = VGroup(*[Line(input.get_right(), output.get_left(), color=GREY_A) for input in hidden2 for output in outputs])
        connections = VGroup(input_hidden1_lines, hidden1_hidden2_lines, hidden2_output_lines)
        net = VGroup(
            inputs, input_hidden1_lines, 
            hidden1, hidden1_hidden2_lines, 
            hidden2, hidden2_output_lines, 
            outputs, 
            input_label, output_label
        ).scale(1/2).move_to(ORIGIN)

        self.add(net)
        self.wait(2)

        input_text = Tex('``To be or not "', color=BLACK).scale(.75).next_to(net, LEFT, buff=.5)
        tokenized_input_text = Tex('(042, 832, 194, 384)', color=BLACK).scale(.65).next_to(net, LEFT, buff=.5)
        output_text = Tex('``to be"', color=RED).scale(.75).next_to(net, RIGHT, buff=.5)
        tokenized_output_text = Tex('(042, 832)', color=RED).scale(.65).next_to(net, RIGHT, buff=.5)

        self.play(FadeIn(input_text, shift=RIGHT))
        self.wait(2)

        self.play(Transform(input_text, tokenized_input_text))
        self.wait(2)

        self.play(ReplacementTransform(input_text, inputs))
        self.wait(2)

        self.play(*(input.animate.set_fill(color=BLACK, opacity=.75) for input in inputs), run_time=.5)
        self.play(
            *(input.animate.set_fill(color=BLACK, opacity=.5) for input in inputs),
            *(line.animate.set_color(GREY_B) for line in input_hidden1_lines), run_time=.5
        )
        self.play(
            *(hidden.animate.set_fill(color=BLUE, opacity=.75) for hidden in hidden1),
            *(line.animate.set_color(GREY_A) for line in input_hidden1_lines), run_time=.5
        )
        self.play(
            *(hidden.animate.set_fill(color=BLUE, opacity=.5) for hidden in hidden1),
            *(line.animate.set_color(GREY_B) for line in hidden1_hidden2_lines), run_time=.5
        )
        self.play(
            *(hidden.animate.set_fill(color=BLUE, opacity=.75) for hidden in hidden2),
            *(line.animate.set_color(GREY_A) for line in hidden1_hidden2_lines), run_time=.5
        )
        self.play(
            *(hidden.animate.set_fill(color=BLUE, opacity=.5) for hidden in hidden2),
            *(line.animate.set_color(GREY_B) for line in hidden2_output_lines), run_time=.5
        )
        self.play(
            *(output.animate.set_fill(color=RED, opacity=.75) for output in outputs),
            *(line.animate.set_color(GREY_A) for line in hidden2_output_lines), run_time=.5
        )
        self.play(*(output.animate.set_fill(color=RED, opacity=.5) for output in outputs), run_time=.5)
        self.wait(2)
        self.play(TransformFromCopy(outputs, tokenized_output_text))
        self.wait(2)
        self.play(Transform(tokenized_output_text, output_text))
        self.wait(2)

        self.play(FadeOut(tokenized_output_text))
        


