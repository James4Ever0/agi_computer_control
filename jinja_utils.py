from log_utils import logger_print

import subprocess

from tempfile import TemporaryDirectory
import black
import jinja2
import shutil
import os
import pyright_utils  # for checking if really installed.
import re

# live share's triple quote issue isn't fixed.

import humps  # default to snake case!


def camelize_with_space(string):
    return humps.camelize(string.replace(" ", "-"))


# ref: https://www.geeksforgeeks.org/python-program-to-convert-camel-case-string-to-snake-case/
def c2s(_str):
    """
    Camel case to snake case.
    """
    # return humps.kebabize(_str).replace("-", "_")
    # res = [_str[0].lower()]
    # for c in _str[1:]:
    #     if c in ("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
    #         res.append("_")
    #         res.append(c.lower())
    #     else:
    #         res.append(c)

    # return "".join(res)
    return humps.decamelize(_str)


def s2c(_str, lower: bool):
    """
    Snake case to camel case.
    """
    # assert not _str.startswith("_")
    # lst = _str.split("_")
    # first_letter = lst[0][0]
    # lst[0] = (first_letter.lower() if lower else first_letter.upper()) + lst[0][1:]
    # for i in range(1, len(lst)):
    #     lst[i] = lst[i].title()
    # return "".join(lst)
    return getattr(humps, "camelize" if lower else "pascalize")(_str)


def s2cl(_str):
    """
    Snake case to camel case (starting with lower letter).
    """
    return s2c(_str, True)


def s2cu(_str):
    """
    Snake case to camel case (starting with upper letter).
    """
    return s2c(_str, False)


class NeverUndefined(jinja2.StrictUndefined):
    def __init__(self, *args, **kwargs):
        # ARGS: ("parameter 'myvar2' was not provided",)
        # KWARGS: {'name': 'myvar2'}
        if len(args) == 1:
            info = args[0]
        elif "name" in kwargs.keys():
            info = f"Undefined variable '{kwargs['name']}"
        else:
            infoList = ["Not allowing any undefined variable."]
            infoList.append(f"ARGS: {args}")
            infoList.append(f"KWARGS: {kwargs}")
            info = "\n".join(infoList)

        raise Exception(info)


def load_render_and_format(
    template_path: str,
    output_path: str,
    render_params: dict,
    banner: str,
    needFormat: bool = True,
):
    tpl = load_template(template_path)
    result = tpl.render(**render_params)

    logger_print()
    logger_print("______________________[{}]".format(banner))
    logger_print(result)

    # import black.Mode
    output_path_elems = output_path.split(".")
    output_path_elems.insert(-1, "new")
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            backup_content = f.read()
    else:
        backup_content = ""
    with open(tmp_output_path := ".".join(output_path_elems), "w+") as f:
        f.write(result)
    if not needFormat:
        shutil.move(tmp_output_path, output_path)
        return
    try:
        # TODO: add more test, like checking for undefined variables, before rewriting the source file.
        # TODO: add rollback mechanism in makefile
        result = black.format_str(result, mode=black.Mode())
        logger_print("Formatter Ok.")
        # with TemporaryDirectory() as TP:
        with open(output_path, "w+") as f:
            f.write(result)
        # do further type checking.

        # typechecker_input_path = os.path.join(
        #     TP, base_output_path := os.path.basename(output_path)
        # )
        # with open(typechecker_input_path, "w+") as f:
        #     f.write(typechecker_input_path)
        # output = subprocess.run(
        #     ["pyright", typechecker_input_path],
        #     capture_output=True,
        #     encoding="utf-8",
        # )
        run_result = pyright_utils.run(
            output_path, capture_output=True, encoding="utf-8"
        )
        typeErrors = [
            e.strip().replace(
                os.path.basename(output_path), os.path.basename(tmp_output_path)
            )
            for e in re.findall(
                pyright_utils.errorRegex, run_result.stdout, re.MULTILINE
            )
        ]
        # breakpoint()
        if run_result.stderr:
            typeErrors.append("")
            typeErrors.append(f"Pyright error:\n{run_result.stderr}")
        if typeErrors:
            typeErrors.insert(0, f"Type error found in file {repr(output_path)}")
            raise Exception(f"\n{' '*4}".join(typeErrors))
        logger_print("Pyright Ok.")
        os.remove(tmp_output_path)
    except:
        import traceback

        traceback.print_exc()
        # os.remove(tmp_output_path)
        with open(output_path, "w+") as f:
            f.write(backup_content)
        # ref: https://www.geeksforgeeks.org/python-os-utime-method/
        # do not set this to 0 or something. will cause error.
        os.utime(
            output_path,
            times=(
                os.path.getatime(template_path) - 1000000,
                os.path.getmtime(template_path) - 1000000,
            ),
        )  # to make this older than template, must update!

        raise Exception(
            f"Code check failed.\nTemporary cache saved to: '{tmp_output_path}'"
        )
    logger_print("=" * 40)


def lstrip(string: str):
    lines = string.split("\n")
    result_lines = []
    for line in lines:
        if stripped_line := line.lstrip():
            result_lines.append(stripped_line)
    result = "\n".join(result_lines)
    return result


def code_and_template_path(base_name):
    code_path = f"{base_name}.py"
    template_path = f"{code_path}.j2"
    return code_path, template_path


def load_template(template_path, extra_func_dict={}):
    try:
        assert template_path.endswith(".j2")
    except:
        Exception(f"jinja template path '{template_path}' is malformed.")
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(searchpath=["./", "../"]),
        extensions=[
            "jinja2_error.ErrorExtension",
            "jinja2.ext.do",
            "jinja2.ext.loopcontrols",
        ],
        trim_blocks=True,
        lstrip_blocks=True,
        # undefined=jinja2.StrictUndefined,
        undefined=NeverUndefined,
    )
    tpl = env.get_template(template_path)
    # def myJoin(mstr, mlist):
    #     logger_print("STR:", repr(mstr))
    #     logger_print("LIST:", repr(mlist))
    #     return mstr.join(mlist)
    func_dict = dict(
        list=list,
        str=str,
        _dict=dict,
        _set=set,  # avoid name collision
        tuple=tuple,
        ord=ord,
        len=len,
        repr=repr,
        c2s=c2s,
        # s2c=s2c,
        s2cl=s2cl,
        s2cu=s2cu,
        zip=zip,
        cws=camelize_with_space,
        lstrip=lstrip,
        # enumerate=enumerate,
        # eval=eval,
        #  join=myJoin
        **extra_func_dict,
    )
    tpl.globals.update(func_dict)
    return tpl


def test(cmd: list, exec="python3" if os.name != "nt" else "python"):
    cmd = [exec] + cmd
    p = subprocess.run(cmd)
    p.check_returncode()
