
spliters = ",.，。！~"

output = 'script.md'
with open('script.txt', 'r') as f:
	data = f.read()

for e in spliters:
	data = data.replace(e, " ")

data = data.strip()

with open(output, 'w+') as f:
	f.write("\n".join(data))
