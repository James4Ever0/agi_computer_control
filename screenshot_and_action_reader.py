from config import filePath, screenshotLogPath

# it may get stuck, by not using vnc protocol.
# it is less efficient.
# but anyway, who cares? this is something we must do, by arranging key events with screenshots.

# though you say i may want to use text-only terminal interfaces, i think there are some commondities in between, so the order doesn't matter so much.
import jsonlines

# import math

# i find the mss is more interesting than obs studio.
# both will stuck when macos gets stuck

from config import timestep
import numpy as np

with jsonlines.open(filePath) as action_logs, jsonlines.open(
    screenshotLogPath
) as screenshot_logs:
    action_list = list(action_logs.iter())
    screenshot_list = list(screenshot_logs.iter())
    # breakpoint()
    # we cannot account for actions going in the dark (without the screenshot), so let's discard these!
    # we slice the timespan which we have screenshots, in which we arrange actions in order, discard the action delay, only preserve the sequence.
    screenshot_start_time, screenshot_end_time = (
        screenshot_list[0]["timeStamp"],
        screenshot_list[-1]["timeStamp"],
    )
    time_slice_points = np.linspace(
        screenshot_start_time,
        screenshot_end_time,
        num=int((screenshot_end_time - screenshot_start_time) / timestep),
    )
    previous_screenshot = None

    screenshot_and_actions_list = []

    for time_slice_start, time_slice_end in zip(
        time_slice_points[:-1], time_slice_points[1:]
    ):
        if screenshot_list == []:
            break
        actions = []
        screenshots = []

        while action_list != []:
            action = action_list.pop(0)
            actionTimeStamp = action["timeStamp"]
            if actionTimeStamp >= time_slice_start and actionTimeStamp < time_slice_end:
                actions.append(action)
            elif actionTimeStamp >= time_slice_end:
                action_list.insert(0, action)
                break
            else:
                continue
        # if no screenshot this time, just use previous screenshot.
        # if previous screenshot is missing, skip this time slice.
        while True:
            screenshot = screenshot_list.pop(0)
            screenshotTimeStamp = screenshot["timeStamp"]
            if (
                screenshotTimeStamp >= time_slice_start
                and screenshotTimeStamp < time_slice_end
            ):
                screenshots.append(screenshot)
            elif screenshotTimeStamp >= time_slice_end:
                screenshot_list.insert(0, screenshot)
                break
            else:
                continue
        if screenshots == []:
            if previous_screenshot:
                current_screenshot = previous_screenshot
            else:
                continue
        else:
            current_screenshot = screenshots[-1].copy()
            previous_screenshot = current_screenshot.copy()

        # we don't care about actions so much. it can always be empty.
        print()
        print("START?", time_slice_start)
        print("END?", time_slice_end)
        print("SCREENSHOT?", current_screenshot)
        print("ACTIONS?", actions)
        screenshot_and_actions_list.append(
            dict(
                screenshot=current_screenshot.copy(),
                actions=actions.copy(),
                timeSlice=dict(start=time_slice_start, end=time_slice_end),
            )
        )

import json
import mss

monitor = mss.mss().monitors[0]
perspective_size = (monitor["width"], monitor["height"])
with open("screenshot_and_actions.json", "w+") as f:
    f.write(
        json.dumps(
            dict(
                screenshot_and_actions=screenshot_and_actions_list,
                perspective_size=perspective_size,
                timestep=timestep,
            ),
            indent=4,
        )
    )
