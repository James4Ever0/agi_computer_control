
spliters = ",.，。！~"

output = 'script.md'
with open('script.txt', 'r') as f:
	data = f.read()

for e in spliters:
	data = data.replace(e, " ")

data = data.strip()
data = data.split(" ")

with open(output, 'w+') as f:
	#f.write("\n".join(data))
	for d in data:
		f.write(f"- text: {d}\n")
		f.write("  video: \n")
