// Create a canvas element to draw the screenshot

var canvas = document.createElement('canvas');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
var ctx = canvas.getContext('2d');

// Draw the screenshot of the viewport onto the canvas
// does not work at all.
ctx.drawImage(window, 0, 0, window.innerWidth, window.innerHeight);
// ctx.drawImage(window, 0, 0, window.innerWidth, window.innerHeight);
// Convert the canvas content to a data URL representing the screenshot
var screenshotDataUrl = canvas.toDataURL('image/png');
canvas.remove()

// Open the data URL in a new tab to display the screenshot
const newTab = window.open();
newTab.document.write('<img src="' + screenshotDataUrl + '" />');
