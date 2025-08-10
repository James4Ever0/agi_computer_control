
# the challenge listing page is in html

curl -Lo vimgolf_public_challenges_page.html "https://www.vimgolf.com/?page=2"

# instead of html, it returns in json format.
# however, in browser, it returns a html webpage.
curl -Lo vimgolf_specific_challenge.json "https://www.vimgolf.com/challenges/9v0067401f2500000000061b"
