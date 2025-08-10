import requests
from parse_asciinema_public_explore_page import parse_asciinema_explore_public_page
from parse_asciinema_cast_description import parse_asciinema_cast_description
import json
import os

# to prevent issue of dns error, just map resolved ip of asciinema.org into /etc/hosts

def test_request_asciinema_explore_pages():
    first_page_url = "https://asciinema.org/explore/public?order=date&page=1"
    nonexistant_page_url = "https://asciinema.org/explore/public?order=date&page=20000"  # this will still return the last page with full content. better find trace in the page.

    response = requests.get(first_page_url)
    print("First page response code:", response.status_code) # 200: first page response code

    response = requests.get(nonexistant_page_url)
    print("Nonexistant page response code:",response.status_code) # 200: nonexistant page response code

def scrape_all_public_explore_pages():
    # page_index = 1
    page_index=4973

    output_jsonl_file = "asciinema_public_explore_pages.jsonl"
    print("Output will be saved at:", output_jsonl_file)

    while True:
        print("Sraping page %s" % page_index)
        url = "https://asciinema.org/explore/public?order=date&page=%s" % page_index
        page_index += 1
        response = requests.get(url)
        print("Status code:", response.status_code)
        assert response.status_code == 200
        html_content = response.text
        iterator = parse_asciinema_explore_public_page(html_content)
        page_metadata = next(iterator)
        print("Page metadata:", repr(page_metadata))
        if type(page_metadata) == dict:
            active_page_num = page_metadata['active_page_num'] # str
            if active_page_num == str(page_index - 1):
                
                with open(output_jsonl_file, "a+") as f:
                    for it in iterator:
                        print(it)
                        f.write(json.dumps(it) + "\n")
            else:
                print("Active page %s is different than current page index %s" % (active_page_num, page_index - 1))
                break
    print("Scraping done")

def download_all_public_cast_and_descriptions():
    input_jsonl_filepath = "asciinema_public_explore_pages.jsonl"
    output_dirpath = "./recordings"

    os.makedirs(output_dirpath, exist_ok=True)

    with open(input_jsonl_filepath, "r") as f:
        for line in f.readlines():
            print("Processing line:", line)
            try:
                data = json.loads(line) # keys: href, title, duration, author, submit_time
            except json.JSONDecodeError:
                print("Skipping line because it is not JSON serializable")
            if type(data) != dict:
                print("Skipping line since data is not of type dict")
            href = data.get("href")
            if href:

                record_id = href.split("/")[-1]
                os.makedirs(f"{output_dirpath}/{record_id}", exist_ok=True)

                description_write_path = f"{output_dirpath}/{record_id}/info.json"

                if not os.path.exists(description_write_path):
                    description_html_url = "https://asciinema.org%s" % href
                    print("Description html url:", description_html_url)

                    print("Downloading description...")
                    description_response = requests.get(description_html_url)
                    description_html = description_response.text

                    description_parse_result = parse_asciinema_cast_description(description_html) # keys: title, description

                    asciicast_file_extension_name = description_parse_result.get('asciicast_file_extension_name')

                    asciicast_version = description_parse_result.get('asciicast_version')

                    data['description'] = description_parse_result.get('description')
                    data['asciicast_version'] = asciicast_version
                    data['asciicast_file_extension_name'] =  asciicast_file_extension_name

                    with open(description_write_path,  'w') as f:
                        f.write(json.dumps(data))
                    print("Write description to file:", description_write_path)
                else:
                    print("Description already exists:", description_write_path)

                with open(description_write_path,  'r') as f:
                    description_data = json.loads(f.read())
                    asciicast_file_extension_name = description_data['asciicast_file_extension_name']

                    cast_write_path = f"{output_dirpath}/{record_id}/record{asciicast_file_extension_name}"

                if not os.path.exists(cast_write_path):
                    cast_url = "https://asciinema.org%s%s?dl=1" % (href, asciicast_file_extension_name)

                    print("Cast download url:", cast_url)

                    print("Downloading asciicast...")

                    cast_response = requests.get(cast_url)
                    cast_bytes = cast_response.content
                    with open(cast_write_path,  'wb') as f:
                        f.write(cast_bytes) 
                    print("Write asciicast to file:", cast_write_path)
                else:
                    print("Asciicast already exists:", cast_write_path)
            else:
                print("Href not found in line, skipping")
                continue

if __name__ == "__main__":
    # test_request_asciinema_explore_pages()
    scrape_all_public_explore_pages() # estimate stop page at 8/10/25: 5667
    # download_all_public_cast_and_descriptions()