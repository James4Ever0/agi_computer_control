import jinja_utils as ju
import sys

target = sys.argv[-1]
print("target output:", target)

assert target.endswith(".py")
basename = target.strip(".py")

code_path, template_path = ju.code_and_template_path(basename)
assert code_path == target

ju.load_render_and_format(template_path, code_path, render_params = {}, banner = basename.replace("_"," ").upper()) # TODO: case by case. ensure we have the right render_params.