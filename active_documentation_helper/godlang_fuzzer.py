import random

special_command_list = [
    "SPACE",
    "BACKSPACE",
    "TAB",
    "ENTER",  # NEWLINE
    "ESC",
    "PGUP",
    "PGDN",
    "END",
    "HOME",
    "UP",
    "META+UP",
    "DOWN",
    "META+DOWN",
    "LEFT",
    "RIGHT",
    "INS",
    "DEL",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "CTRL+A",
    "CTRL+B",
    "CTRL+C",
    "CTRL+D",
    "CTRL+E",
    "CTRL+F",
    "CTRL+G",
    "CTRL+H",
    "CTRL+I",
    "CTRL+J",
    "CTRL+K",
    "CTRL+L",
    "CTRL+M",
    "CTRL+N",
    "CTRL+O",
    "CTRL+P",
    "CTRL+Q",
    "CTRL+R",
    "CTRL+S",
    "CTRL+T",
    "CTRL+U",
    "CTRL+V",
    "CTRL+W",
    "CTRL+X",
    "CTRL+Y",
    "CTRL+Z",
    "CTRL+0",
    "CTRL+1",
    "CTRL+2",
    "CTRL+3",
    "CTRL+4",
    "CTRL+5",
    "CTRL+6",
    "CTRL+7",
    "CTRL+8",
    "CTRL+9",
]

import string

additional_special_command_list = [
    "NEWLINE",  # '\n'
    "CARRIAGE_RETURN",
    "CR",
    "CRETURN",
    "VERTICAL_TAB",
    "VTAB",
    "FORM_FEED",
    "FF",
]

# TODO: should you put these into special command list
invisible_chars = [
    it for it in string.whitespace if it != "\n"
]  # Invisible characters, including '\r'

char_list = list(string.printable) + invisible_chars


class BaseCommandGenerator:
    @classmethod
    def call_single_random_command(cls):
        command_generator_method_name = random.choice(
            [it for it in dir(cls) if it.startswith("get_")]
        )
        command: str = getattr(cls, command_generator_method_name)()
        return command


class CommandGenerator(BaseCommandGenerator):
    @staticmethod
    def get_random_chars():
        random_chars = "".join(random.sample(char_list, k=random.randint(1, 20)))
        return f'TYPE {random_chars}'
        # return random_chars

    @staticmethod
    def get_special_command():
        special_command = random.choice(special_command_list)
        return f"SPECIAL {special_command}"
        # return special_command

    # no special command escaping.
    
    # @staticmethod
    # def get_type_command():
    #     special_command = CommandGenerator.get_special_command()
    #     return f"TYPE {special_command}"

    @staticmethod
    def get_view_command():
        return "VIEW"

    @staticmethod
    def get_wait_command():
        wait_time = random.uniform(0.1, 1)
        return f"WAIT {wait_time:.3f}"

    @staticmethod
    def get_reminder_command(reminder="Random actions"):
        return f"REM {reminder}"


if __name__ == "__main__":
    for _ in range(100):
        command = CommandGenerator.call_single_random_command()
        print(command)
