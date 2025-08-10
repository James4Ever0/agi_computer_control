from bs4 import BeautifulSoup

html_filepath = "vimgolf_public_challenges_page.html"

with open(html_filepath, 'r') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')

