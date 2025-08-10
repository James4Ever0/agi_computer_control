enumerate asciinema.org terminal recordings by page:

https://asciinema.org/explore/public?order=date&page=2

visit a specific recording:

https://asciinema.org/a/648882

group recordings into v1, v2, v3

download (v2) cast:

https://asciinema.org/a/648882.cast?dl=1

the plan:

- iterate incrementally from page index 1 to infinity, stop when the active page number is less than the page index
- append all recording info into a jsonl file "public_asciinema_recordings.jsonl"
- iterate through the jsonl file, download each recording cast file and detail info, merge the detail info with the one in jsonl file
- store the recording cast files in a directory "recordings/<recording_id>/record.cast"
- store the recording detail info in a directory "recordings/<recording_id>/info.json"
