<!-- TODO: type annotate all client library api return values, for ctf and cybergod (just json parse those these things, create pydantic types) -->
<!-- TODO: write those return values of api in readme code snippets -->
<!-- format parsing code style: first log the input, then parse it with the success return format, if not success, just return nothing, or just using true and false for those non info retrieval tasks -->

Cybergod learning environment primitives, including:

- ctf: Capture the flag
- cybergod: Central bank and crypto currency financial system with chronic taxs

The agent is required to use multiple computer environments for solving a specific challenge, including vimgolf. For example, one GUI environment with Firefox browser for looking up vim cheatsheet online, and one terminal environment for interacting with vimgolf challenge program.

The agent is encouraged to use the secondary environment for information collection and may not count into the main environment steps, depending on the game policy.

Example code:

```python
with gui_firefox_env:
    with terminal_vimgolf_env:
        gui_firefox_env.step("pynput.mouse.Controller().position = (1, 1)")
        gui_screenshot = gui_firefox_env.screenshot()
        terminal_vimgolf_env.step("i")
        terminal_screenshot = terminal_vimgolf_env.screenshot()
```
