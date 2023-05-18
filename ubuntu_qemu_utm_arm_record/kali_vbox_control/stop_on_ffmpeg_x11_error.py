error = "Failed to query xcb pointer"
while True:
    data = input()
    print(data)
    if error in data:
        break