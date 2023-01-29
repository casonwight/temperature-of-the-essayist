from manim import *


ASSET_PATH = "animation/assets/"

class Transformer(Scene):
    def construct(self):
        self.camera.background_color = WHITE

        # Words to ingest are in the middle of the screen
        text = r"$\textlangle\text{SOS}\textrangle$ To be or not".split()
        words = VGroup(*[Tex(word, color=GREY_A) for word in text]).arrange(RIGHT, buff=3/4).move_to(ORIGIN).shift(UP + LEFT * 2)
        for word in words[1:]:
            word.align_to(words[0], DOWN)

        # Encoder will appear below the first word
        encoder = RoundedRectangle(height=1, width=2, corner_radius=0.25, color=BLUE).set_fill(opacity=0.5)  
        encoder.scale(3/4).next_to(words[0], DOWN)

        # Encoder has gears in it
        gear = SVGMobject(ASSET_PATH + "gear.svg").set_fill(color=GREY, opacity=1)
        encoder_gears = VGroup(
            gear.copy().shift(LEFT+UP),
            gear.copy().shift(RIGHT),
            gear.copy().scale(3/4).shift(LEFT * 3/4 + DOWN),
        ).scale(1/8).move_to(encoder.get_center())

        # Decoder is below the encoder
        decoder = RoundedRectangle(height=2, width=4, corner_radius=0.25, color=PURPLE).set_fill(opacity=0.5).scale(3/4).next_to(encoder, DOWN).align_to(encoder, LEFT)

        # Decoder has gears
        decoder_gears = encoder_gears.copy().align_to(decoder, UR).shift(LEFT * 1/8 + DOWN * 1/8)
        
        # Decoder will have words
        decoded_text = r"To be or not to be".split()
        decoded_words = VGroup(*[Tex(word, color=BLACK) for word in decoded_text]).scale(1/2).arrange(RIGHT, buff=1/8).set_opacity(0)
        for word in decoded_words[1:]:
            word.align_to(decoded_words[0], DOWN)
        decoded_words.move_to(decoder.get_center())
        decoded_words.add(Tex(r"$\textlangle\text{EOS}\textrangle$").set_opacity(0))
        
        # Decoder will have label
        decoder_label = Tex("Decoder", color=BLACK).scale(1/2).next_to(decoder, LEFT)

        # Encoder will have label
        encoder_label = Tex("Encoder", color=BLACK).scale(1/2).add_updater(lambda x: x.next_to(encoder, LEFT))

        # The whole thing moves together as a transformer
        transformer = VGroup(
            encoder,
            encoder_gears,
            decoder,
            decoder_gears,
            encoder_label,
            decoder_label,
            decoded_words
        )
        
        self.play(Write(words))
        self.wait()

        self.play(
            FadeIn(transformer, shift=RIGHT),
            words[0].animate.set_color(BLACK)
        )
        self.wait()
        shift_dist = (words[1].get_center() - encoder.get_center())[0]

        self.play(
            *(Rotate(gear, angle=(int(i % 2 == 1) * 2 - 1) * PI, about_point=gear.get_center(), rate_func=linear) for i, gear in enumerate(encoder_gears)),
            words[0].animate.move_to(encoder_gears.get_center()).scale(0),
        )  
        self.remove(words[0])
        self.play(*(Rotate(gear, angle=(int(i % 2 == 1) * 2 - 1) * PI, about_point=gear.get_center(), rate_func=linear) for i, gear in enumerate(encoder_gears)))

        for word, next_word, decoded_word in zip(words[1:], list(words[2:]) + [words[-1]], decoded_words[:-2]):
            self.play(
                transformer.animate.shift(RIGHT * shift_dist),
                word.animate.set_color(BLACK),
            )
            self.wait()
            
            shift_dist = (next_word.get_center() - encoder.get_center())[0]
            
            self.play(
                *(Rotate(gear, angle=(int(i % 2 == 1) * 2 - 1) * PI, about_point=gear.get_center(), rate_func=linear) for i, gear in enumerate(encoder_gears)),
                word.animate.move_to(encoder_gears.get_center()).scale(.01),
            )
            self.play(
                *(Rotate(gear, angle=(int(i % 2 == 1) * 2 - 1) * PI, about_point=gear.get_center(), rate_func=linear) for i, gear in enumerate(encoder_gears)),
                *(Rotate(gear, angle=(int(i % 2 == 1) * 2 - 1) * PI, about_point=gear.get_center(), rate_func=linear) for i, gear in enumerate(decoder_gears)),
                ReplacementTransform(word, decoded_word),
                decoded_word.animate.set_opacity(1),
            )
            self.remove(word)
            self.wait()

        
        pred_words = [
            ["today (5\%)", "to (93\%)", r"$\textlangle\text{EOS}\textrangle$ (2\%)"],
            ["be (94\%)", "think (3\%)", r"$\textlangle\text{EOS}\textrangle$ (3\%)"],
            ["not (1\%)", ". (40\%)", r"$\textlangle\text{EOS}\textrangle$ (59\%)"]
        ]

        correct_words = [1, 0, 2]
        incorrect_words = [
            [0, 2],
            [1, 2],
            [0, 1]
        ]

        for i, (correct, incorrect, poss_words, decoded_word) in enumerate(zip(correct_words, incorrect_words, pred_words, decoded_words[-3:])):
            correct_word = Tex(poss_words[correct], color=BLACK).scale(1/2)
            incorrect_word_1 = Tex(poss_words[incorrect[0]], color=BLACK).scale(1/2)
            incorrect_word_2 = Tex(poss_words[incorrect[1]], color=BLACK).scale(1/2)
            pred_words = VGroup(incorrect_word_1, correct_word, incorrect_word_2).arrange(DOWN).next_to(decoder, RIGHT)

            incorrect_word_1.align_to(correct_word, LEFT)
            incorrect_word_2.align_to(correct_word, LEFT)

            self.play(
                *(Rotate(gear, angle=(int(i % 2 == 1) * 2 - 1) * PI, about_point=gear.get_center(), rate_func=linear) for i, gear in enumerate(decoder_gears)),
                TransformFromCopy(decoded_words, pred_words)
            )
            self.wait()

            self.play(
                FadeOut(incorrect_word_1),
                FadeOut(incorrect_word_2),
            )

            self.play(
                correct_word.animate.move_to(encoder_gears.get_center()).scale(.05),
                *(Rotate(gear, angle=(int(i % 2 == 1) * 2 - 1) * PI, about_point=gear.get_center(), rate_func=linear) for i, gear in enumerate(encoder_gears)),
                
            )
            if i < 2:
                self.play(
                    *(Rotate(gear, angle=(int(i % 2 == 1) * 2 - 1) * PI, about_point=gear.get_center(), rate_func=linear) for i, gear in enumerate(encoder_gears)),
                    *(Rotate(gear, angle=(int(i % 2 == 1) * 2 - 1) * PI, about_point=gear.get_center(), rate_func=linear) for i, gear in enumerate(decoder_gears)),
                    Transform(correct_word, decoded_word),
                    decoded_word.animate.set_opacity(1),
                )
            else:
                self.play(
                    *(Rotate(gear, angle=(int(i % 2 == 1) * 2 - 1) * PI, about_point=gear.get_center(), rate_func=linear) for i, gear in enumerate(encoder_gears)),
                    FadeOut(correct_word)
                )
    

        self.play(
            FadeOut(transformer, shift=RIGHT)
        )
