from bs4 import BeautifulSoup

html_filepath = "vimgolf_public_challenges_page.html"

with open(html_filepath, 'r') as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, 'html.parser')

for h5_elem in soup.find_all("h5", class_="challenge"):
    # print("H5 element:", h5_elem)
    a_href_elem = h5_elem.find("a")
    href = a_href_elem['href']
    title = a_href_elem.text
    p_elem = h5_elem.find_next_sibling("p")
    detail = p_elem.text
    print()
    print("Href:", href)
    print("Title:", title)
    print("Detail:", detail)