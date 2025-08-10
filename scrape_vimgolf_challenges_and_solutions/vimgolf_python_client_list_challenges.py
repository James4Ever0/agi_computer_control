import subprocess
import parse

page_1_challenges_output = subprocess.check_output("vimgolf list 1".split()).decode()

# print("Page 1 challenges:")

# print(page_1_challenges_output)

for line in page_1_challenges_output.splitlines():
    if line.startswith("+"):
        print("Parsing line:", line)
        line = line.strip()
        result = parse.parse('+{num} {title} - {entries_count} entries ({hash})', line)
        print("Parse result:", result)
    else:
        print("Skipping line:", line)
