from bs4 import BeautifulSoup

def test():
    html_filepath = "asciinema_cast_description.html"
    with open(html_filepath, "r") as f:
        html_doc = f.read()
    parse_asciinema_cast_description(html_doc)

def parse_asciinema_cast_description(html_doc:str):
    soup = BeautifulSoup(html_doc, 'html.parser')

    section_even_info = soup.find("section", class_="even info")

    if section_even_info:
    # if False: # test failsafe branch
        title_h2 = section_even_info.find("h2")
        print("Title:")
        if hasattr(title_h2, "text"):
            title_text = title_h2.text # type: ignore
        else:
            title_text = "No title found"
        print(title_text)
    else:
        print("Section even_info not found")
        # instead, find the title text directly from header.
        title_element = soup.find("title")
        print("Title:")
        title_suffix = "- asciinema.org"
        if title_element:
            title_text = title_element.text.replace(title_suffix, "")
        else:
            title_text = "No title found"
        print(title_text)


    description_div = soup.find("div", class_ = "description")
    print("Description:") 
    if description_div:
        description_text = description_div.text
    else:
        description_text = "No description found"
    print(description_text)

    asciicast_file_extension_name = ".json" # v1 extension: .json; v2 extension: .cast

    asciicast_extension_element = soup.find(id='download-link')
    if asciicast_extension_element:
        print("Found div#download-link")
        asciicast_extension_strong_element = asciicast_extension_element.find("strong")
        if asciicast_extension_strong_element:
            print("Found asciicast extension element in HTML")
            asciicast_file_extension_name = asciicast_extension_strong_element.text # type: ignore
    
    print("Asciicast file extension name:", asciicast_file_extension_name)

    asciicast_version = "unknown"

    download_modal_element = soup.find(id="download-modal")
    if download_modal_element:
        print("Found div#download-modal")
        asciicast_version_element = download_modal_element.find("div", class_="modal-body")
        if asciicast_extension_element:
            print("Found asciicast version element in HTML")
            if "asciicast v2 format" in asciicast_version_element.text:
                asciicast_version = "v2"
            elif "asciicast v1 format" in asciicast_version_element.text:
                asciicast_version = "v1"
            elif "asciicast v3 format" in asciicast_version_element.text:
                asciicast_version = "v3"
    
    print("Asciicast version:", asciicast_version)
    
    # TODO: views, to be added into info
    span_view_element = soup.find("span", attrs=dict(title="Total views"))
    total_views="unknown"
    if span_view_element:
        print("Span view element found")
        total_views = span_view_element.text.strip()
        total_views = total_views.split()[0]
    else:
        print('Span view element not found')
    print("Total views:", total_views)

    return dict(title=title_text, description=description_text, asciicast_file_extension_name=asciicast_file_extension_name, asciicast_version=asciicast_version)

if __name__ == "__main__":
    test()
