from bs4 import BeautifulSoup

with open("vimgolf_challenge.html", "r") as f:
    content = f.read()

soup = BeautifulSoup(content, features='lxml')
elem_list= soup.select(".success > pre:nth-child(2)")

elem = elem_list[0]

text = elem.text
print(text.strip())

# TODO: translate this into godlang.