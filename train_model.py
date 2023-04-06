# try to train a model.
# first let's get the data.

# keep it simple, just want to scratch the surface.
# we can run this on cpu.

import json

with open("screenshot_and_actions.json", "r") as f:
    data = json.loads(f.read())

# print(data.keys())
# ['screenshot_and_actions', 'perspective_size', 'timestep']

perspective_width, perspective_height = data["perspective_size"]
timestep = data["timestep"]

print("PERSPECTIVE:", perspective_width, perspective_height)
print("TIMESTEP:", timestep)

# you could embed the image size into the imagePath.

for screenshot_and_actions in data["screenshot_and_actions"]:
    screenshot = screenshot_and_actions["screenshot"]
    actions = screenshot_and_actions["actions"]
    print()
    print("SCREENSHOT:", screenshot)
    # SCREENSHOT: {'timeStamp': 1680247598.852561, 'imagePath': 'screenshots/498.raw', 'imageSize': [2560, 1600]}
    print("ACTIONS:", actions)
    # keyboard action types: ['key_press', 'key_release']
    # mouse action types: ['mouse_move', 'mouse_click', 'mouse_scroll']
    # [{'HIDEvent': ['mouse_move', [854.43359375, 488.60546875]], 'timeStamp': 1680247553.570472}, {'HIDEvent': ['mouse_move', [850.5859375, 491.0078125]], 'timeStamp': 1680247553.5704901}]

    # now, try to map these actions to neural networks.
    # let's see some example?
