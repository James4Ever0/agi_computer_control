
pull prebuilt image:

docker pull agile4im/cybergod_worker_gui:remote-base

docker tag agile4im/cybergod_worker_gui:remote-base cybergod_worker_gui:remote-base

---

can start a new gui program at :1 without window manager

openssh-client is already installed in x11vnc/docker-desktop

install tools:

xdotool - xdotool
x11-utils - xprop

tigervnc-viewer - vncviewer

tightvncpasswd - tightvncpasswd

https://manpages.debian.org/stretch/tigervnc-common/tigervncpasswd.1.en.html

remmina
remmina-plugin-vnc

wm classes:

lxterminal - lxterminal
"TigerVNC Viewer" - vncviewer

---

tigervnc self connection:

vncviewer -passwd /home/ubuntu/.vnc/passwd0 localhost:5900

---

To achieve a kiosk-like fullscreen mode in LXDE (which uses Openbox as its window manager) with no title bar, disabled resizing/moving, and no window controls, follow these steps:

### 1. **Identify the Application's Window Class**
   - Launch the application.
   - Open a terminal and run `xprop`, then click on the application's window.
   - Note the `WM_CLASS(STRING)` value (e.g., `"myapp"` or `"MyAppClass"`).  
     *(Example output: `WM_CLASS(STRING) = "chromium", "Chromium"`)*

---

### 2. **Configure Openbox Window Rules**
   Edit the Openbox configuration file (specific to LXDE):
   ```bash
   nano ~/.config/openbox/lxde-rc.xml
   ```

   Add this **`<application>` rule** inside the `<applications>` section:
   ```xml
   <application class="YourAppClass" type="normal">
     <!-- Remove title bar and window buttons -->
     <decor>no</decor>
     
     <!-- Force fullscreen -->
     <fullscreen>yes</fullscreen>
     
     <!-- Prevent moving/resizing -->
     <focus>yes</focus>
     <position force="yes">
       <x>0</x>
       <y>0</y>
     </position>
     <size>
       <width>100%</width>
       <height>100%</height>
     </size>
     <maximized>true</maximized>
   </application>
   ```
   - Replace `YourAppClass` with the **second string** from `WM_CLASS` (e.g., `"Chromium"`).

---

### 3. **Disable the Escape Key (and Other Keys)**
   Use **Openbox keybindings** to block keys globally:
   ```xml
   <keybind key="Escape">
     <action name="Execute">
       <command>true</command>  <!-- Do nothing -->
     </action>
   </keybind>
   ```
   Place this in the `<keyboard>` section of `lxde-rc.xml`.  
   *(Optional: Block other keys like `Alt+F4`, `Ctrl+W`, etc., using similar rules.)*

---

### 4. **Restart Openbox**
   Apply changes without logging out:
   ```bash
   openbox --restart
   ```

---

### 5. **Launch the Application**
   Start your app normally. Openbox will enforce the rules.

---

### Alternative: Use `xdotool` for Dynamic Control (Optional)
If rules don’t apply immediately, force settings on startup with a script:
```bash
#!/bin/bash
your_app &  # Launch your app
sleep 2     # Wait for the window to appear

# Get window ID
WIN_ID=$(xdotool search --class "YourAppClass" | head -1)

# Remove decorations
xprop -id $WIN_ID -format _MOTIF_WM_HINTS 32c -set _MOTIF_WM_HINTS "0x2, 0x0, 0x0, 0x0, 0x0"

# Force fullscreen
xdotool windowactivate --sync $WIN_ID windowsize $WIN_ID 100% 100% set_window --overrideredirect 1
```
- Run this script instead of launching the app directly.

---

### For Chromium Specifically
Use **built-in kiosk mode** (bypasses the need for window manager tweaks):
```bash
chromium-browser --kiosk --incognito --noerrdialogs --disable-features=TranslateUI \
  --disable-pinch --overscroll-history-navigation=0 --app=http://your-url
```

---

### Notes:
- **Blocking Keys**: Disabling `Escape` via Openbox affects **all applications**. For app-specific key blocking, tools like `xkeysnail` (advanced) are needed.
- **LXDE Session**: Ensure no other window managers are running (e.g., `compiz` might override Openbox).
- **X11 Overrides**: Tools like `devilspie2` can also manage window properties but are redundant with Openbox rules.

This method uses LXDE/Openbox’s native features to create a restricted kiosk environment without extra tools.