import requests
from parse_vimgolf_public_challenges import parse_vimgolf_public_challenges
import json
import os

def test_request_pages():
    second_page_url = "https://www.vimgolf.com/?page=2"
    nonexistant_page_url = "https://www.vimgolf.com/?page=20"

    second_page_response = requests.get(second_page_url)
    print("Second page response status code:", second_page_response.status_code)
    print("Second page response length:", len(second_page_response.text))


    nonexistant_page_response = requests.get(nonexistant_page_url)
    print("Nonexistant page response status code:", nonexistant_page_response.status_code) # code is 200
    print("Nonexistant page response length:", len(nonexistant_page_response.text))
    print(nonexistant_page_response.text) # still returns a large amount bf data, but the challenge list is empty. we can use that as an indicator.

def scrape_public_challenge_pages():
    page_index = 1
    output_jsonl_path = "vimgolf_public_challenges.jsonl"
    while True:
        print("Scraping page", page_index)
        url = f"https://www.vimgolf.com/?page={page_index}"
        page_index +=1
        response = requests.get(url)
        challenge_items = list(parse_vimgolf_public_challenges(response.text))
        if len(challenge_items) == 0:
            print("Break loop since no challenges found in page")
            break
        with open(output_jsonl_path, 'a+') as f:
            for it in challenge_items:
                f.write(json.dumps(it)+"\n")
    print("Scraping challenge pages done")

def download_all_challenges():
    input_jsonl_path = "vimgolf_public_challenges.jsonl"
    output_dir = "./challenges"
    with open(input_jsonl_path, 'r') as f:
        for line in f.readlines():
            data = json.loads(line)
            href = data['href']
            url = "https://vimgolf.com%s" % href
            challenge_hash = href.split("/")[-1]
            data['challenge_hash'] = challenge_hash
            challenge_savedir= os.path.join(output_dir, challenge_hash)
            os.makedirs(challenge_savedir, exist_ok=True)
            challenge_json_savepath = os.path.join(challenge_savedir, "challenge.json")
            metadata_json_savepath = os.path.join(challenge_savedir, "metadata.json")
            with open(metadata_json_savepath, 'w+') as f:
                f.write(json.dumps(data))
            print("saving metadata at:", metadata_json_savepath)
            if os.path.exists(challenge_json_savepath):
                print("Challenge file %s exists" % challenge_json_savepath)
                print("Skipping download challenge %s" % challenge_hash)
                continue

            response = requests.get(url)
            print("Downloading challenge", url)
            with open(challenge_json_savepath, 'w+') as f:
                f.write(response.text)
            print("saving challenge data at:", challenge_json_savepath)

if __name__ == "__main__":
    # test_request_pages()
    # scrape_public_challenge_pages()
    download_all_challenges()