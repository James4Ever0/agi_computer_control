"use strict";

// when recording, emit action code, timestamp
// generate the time to WAIT later, deduced from timestamp
// you may not know the time for VIEW command. you may do that periodically
// it might be unnecessary to use "TYPE" command.
// paste action might be equivalent to "TYPE" command
// REM statements can be manually added later.

// TODO: handle paste events (CTRL+SHIFT+V)
// TODO: handle copy events (CTRL+SHIFT+C)

function dumpEventData(e) {
    let event_keys = Object.keys(e.__proto__);
    let event_data = {}
    for (let k of event_keys) {
        event_data[k] = e[k];
    }
    return event_data;
}

const encoder = new TextEncoder();
window.onload = () => {
    let terminal = new Terminal("screen", 80, 24);

    let socket;
    function connect() {
        socket = new WebSocket(`ws://${window.location.host}/ws`);
        socket.onmessage = (e) => {
            let msg = JSON.parse(e.data);
            if (msg.type == "update") { terminal.update(msg.data) }
            else if (msg.type == "identifier") { document.title = `Terminal: ${msg.data}` }
            else {
                alert("unknown message from server:", e.data);
            }
        };
        // socket.onclose = () => setTimeout(connect, 5000) // no reconnection!
        socket.onclose = () => { console.log("Terminal closed."); alert("Terminal has been closed."); window.close(); }
    }

    connect();

    const element = document.getElementById("terminal");
    element.onkeydown = e => {
        console.log("keydown", dumpEventData(e))
        let message = keyToMessage(e);
        let action = keyToAction(e);
        let timestamp = Date.now();
        console.log("message (keydown)", message)
        if (message !== null) {
            console.log("<keydown> ACTION:", action, "MESSAGE:", message, "TIMESTAMP:", timestamp)
            socket.send(message);
            socket.send(encoder.encode(JSON.stringify({ action: action, timestamp: timestamp, message: message })))
            e.preventDefault();
            return false;
        }
    };
    element.onkeypress = e => {
        console.log("keypress", dumpEventData(e))
        // let action_and_message = keyToMessage(e);
        let action = keyToAction(e)
        // let action = action_and_message[0]
        let message = keyToMessage(e)
        // let message = action_and_message[1]
        let timestamp = Date.now();
        // let message = keyToMessage(e);
        console.log("message (keypress)", message)
        if (message !== null) {
            console.log("<keypress> ACTION:", action, "MESSAGE:", message, "TIMESTAMP:", timestamp)
            socket.send(message);
            socket.send(encoder.encode(JSON.stringify({ action: action, timestamp: timestamp, message: message })))

        }
    };
    element.focus()
};

class Terminal {
    constructor(id, width, height) {
        this.width = width;
        this.height = height;
        this.screen = new Array(height);
        this.cursor = { x: 0, y: 0 };

        const table = document.getElementById(id);
        for (let i = 0; i < height; i++) {
            this.screen[i] = new Array(width);
            const row = table.insertRow(i);
            for (let j = 0; j < width; j++) {
                const char = row.insertCell(j);
                char.id = `char_${i}_${j}`;
                char.className = "fg-default bg-default";
                char.innerText = " ";
                this.screen[i][j] = char;
            }
        }
    }

    charAt(i, j) {
        return document.getElementById(`char_${i}_${j}`);
    }

    update({ c: [cx, cy], lines }) {
        // Manual line feed hack. ``pyte.Screen`` does that lazily.
        if (cx == this.width) {
            cx = 0;
            cy++;
        }

        this.eraseCursor();

        for (let [i, line] of lines) {
            for (let j = 0; j < this.width; j++) {
                const [data, reverse, fg, bg] = line[j];
                const char = this.charAt(i, j);
                char.innerText = data;

                if (reverse) {
                    this.setStyle(char, bg, fg);
                } else {
                    this.setStyle(char, fg, bg);
                }
            }
        }

        this.updateCursor(cx, cy);
    }

    setStyle(char, fg, bg) {
        if (char === null) {
            return;  // Ignore out of bounds chars.
        }

        char.className = char.className
            .replace(new RegExp(`fg-\\w+`), "fg-" + fg)
            .replace(new RegExp(`bg-\\w+`), "bg-" + bg);
    }

    updateCursor(x, y) {
        this.cursor = { x, y };
        this.setStyle(this.charAt(y, x), "black", "white");
    }

    eraseCursor() {
        const { x, y } = this.cursor;
        this.setStyle(this.charAt(y, x), "default", "default");
    }
}

function keyToMessage(e) {
    if (e.type === "keypress") {
        if (e.which !== 0 && e.charCode !== 0) {
            return (e.which < 32)
                ? null  // special symbol.
                : String.fromCharCode(e.which);
        }

        return null;    // special symbol.
    }

    console.assert(e.type === "keydown") // keydown is for special keys.
    let message = null;
    switch (e.which) {
        case 8:
            message = BACKSPACE; break;
        case 9:
            message = TAB; break;
        case 13:  // Enter
            message = "\n"; break;
        case 27:
            message = ESC; break;
        case 33:  // PgUp
            message = CSI + "5~"; break;
        case 34:  // PgDn
            message = CSI + "6~"; break;
        case 35:  // End
            message = CSI + "4~"; break;
        case 36:  // Home
            message = CSI + "1~"; break;
        case 37:  // Left
            message = CSI + "D"; break;
        case 38:  // Up
            message = e.metaKey ? ESC + "P" : CSI + "A"; break;
        case 39:  // Right
            message = CSI + "C"; break;
        case 40:  // Down
            message = e.metaKey ? ESC + "N" : CSI + "B"; break;
        case 45:  // INS
            message = CSI + "2~"; break;
        case 46:  // DEL
            message = CSI + "3~"; break;
        default:
            if (e.which >= 112 && e.which <= 123) {  // F1 -- F12
                let number = e.which - 111;
                message = F_N[number];
            } else if (e.ctrlKey) {  // Ctrl + ...
                if (e.keyCode >= 65 && e.keyCode <= 90) {     // keycode in A..Z
                    message = String.fromCharCode(e.keyCode - 65 + 1);
                } else if (e.which >= 48 && e.which <= 57) {  // keycode in 0..9
                    message = CONTROL_N[e.which - 48];
                }
            }
    }

    return message;
}


function keyToAction(e) {
    if (e.type === "keypress") {
        if (e.which !== 0 && e.charCode !== 0) {
            return (e.which < 32)
                ? null  // special symbol.
                : 'TYPE';
        }

        return null;    // special symbol.
    }

    console.assert(e.type === "keydown") // keydown is for special keys.
    let message = null;
    switch (e.which) {
        case 8:
            message = "BACKSPACE"; break;
        case 9:
            message = "TAB"; break;
        case 13:  // Enter
            message = "ENTER"; break;
        case 27:
            message = "ESC"; break;
        case 33:  // PgUp
            message = "PGUP"; break;
        case 34:  // PgDn
            message = "PGDN"; break;
        case 35:  // End
            message = "END"; break;
        case 36:  // Home
            message = "HOME"; break;
        case 37:  // Left
            message = "LEFT"; break;
        case 38:  // Up
            message = e.metaKey ? "META+UP" : "UP"; break;
        case 39:  // Right
            message = "RIGHT"; break;
        case 40:  // Down
            message = e.metaKey ? "META+DOWN" : "DOWN"; break;
        case 45:  // INS
            message = "INS"; break;
        case 46:  // DEL
            message = "DEL"; break;
        default:
            if (e.which >= 112 && e.which <= 123) {  // F1 -- F12
                let number = e.which - 111;
                message = "F" + number;
            } else if (e.ctrlKey) {  // Ctrl + ...
                if (e.keyCode >= 65 && e.keyCode <= 90) {     // keycode in A..Z
                    message = "CTRL" + String.fromCharCode(e.keyCode);
                } else if (e.which >= 48 && e.which <= 57) {  // keycode in 0..9
                    message = "CTRL" + (e.which - 48);
                }
            }
    }

    return message;
}

const BACKSPACE = "\u0008";
const TAB = "\u0009";
const ESC = "\u001b";
const CSI = ESC + "[";

// Fdigit
const F_N = {
    1: CSI + "[A",
    2: CSI + "[B",
    3: CSI + "[C",
    4: CSI + "[D",
    5: CSI + "[E",
    6: CSI + "17~",
    7: CSI + "18~",
    8: CSI + "19~",
    9: CSI + "20~",
    10: CSI + "21~",
    11: CSI + "23~",
    12: CSI + "24~"
};

// Ctrl + ....
const CONTROL_N = {
    0: "\u0030",
    1: "\u0031",
    2: "\u0000",
    3: "\u001b",
    4: "\u001c",
    5: "\u001d",
    6: "\u001e",
    7: "\u001f",
    8: "\u007f",
    9: "\u0039",
};