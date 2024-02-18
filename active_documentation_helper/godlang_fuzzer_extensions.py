# extensions for ducky scripts, mouse controls and touchpad controls. (no stylus? midi?)
# anyway, we are building this from language level, independent of the implementation.

from godlang_fuzzer import BaseCommandGenerator, random, special_command_list, char_list


class CoordinateGenerator:
    @staticmethod
    def generate_dxdy():
        dx = random.randint(0, 20)
        dy = random.randint(0, 20)
        return dx, dy

    @staticmethod
    def generate_xy():
        x = random.randint(0, 1080)
        y = random.randint(0, 1920)
        return x, y


class DuckyGenerator(BaseCommandGenerator):  # TODO: implement more ducky script syntax
    @staticmethod
    def generate_key():  # TODO: watch out the space!
        return random.choice(special_command_list + list(char_list))

    @staticmethod
    def get_hold_command():
        key = DuckyGenerator.generate_key()
        return f"HOLD {key}"

    @staticmethod
    def get_release_command():
        key = DuckyGenerator.generate_key()
        return f"RELEASE {key}"

    @staticmethod
    def get_reset_command():
        return "RESET"


class MouseGenerator(
    BaseCommandGenerator, CoordinateGenerator
):  # shall you specify the screen size before fuzzing?
    @staticmethod
    def generate_mouse_button():
        mouse_button = random.choice(["LEFT", "MIDDLE", "RIGHT"])
        return mouse_button

    @staticmethod
    def get_moveto_command():
        x, y = MouseGenerator.generate_xy()
        return f"MOVETO {x} {y}"

    @staticmethod
    def get_relmove_command():
        dx, dy = MouseGenerator.generate_dxdy()
        return f"RELMOVE {dx} {dy}"

    @staticmethod
    def get_click_command():
        mouse_button = MouseGenerator.generate_mouse_button()
        return f"CLICK {mouse_button}"

    @staticmethod
    def get_hold_command():
        mouse_button = MouseGenerator.generate_mouse_button()

        return f"HOLD {mouse_button}"

    @staticmethod
    def get_release_command():
        mouse_button = MouseGenerator.generate_mouse_button()

        return f"RELEASE {mouse_button}"

    @staticmethod
    def get_scroll_command():
        dx, dy = MouseGenerator.generate_dxdy()
        return f"SCROLL {dx} {dy}"


class TouchpadGenerator:
    ...


class StylusGenerator:
    ...


class MIDIGenerator:
    ...


class MetaGenerator:
    def __init__(self, generator_class_list: list[BaseCommandGenerator]):
        self.generator_class_list = generator_class_list

    def generate_command(self):
        generator_class = random.choice(self.generator_class_list)
        command = generator_class.call_single_random_command()
        return command
