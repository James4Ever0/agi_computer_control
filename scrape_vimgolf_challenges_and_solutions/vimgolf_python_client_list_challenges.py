import subprocess
import parse

page_1_challenges_output = subprocess.check_output("vimgolf list 1")

print("Page 1 challenges:")
print(page_1_challenges_output)
