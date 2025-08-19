const TerminalRenderer = require('./terminal-to-png');
const fs = require('fs');

async function main() {
  const renderer = new TerminalRenderer({
    cols: 80,
    rows: 25,
    fontSize: 14
  });

  // Simulate terminal output
  renderer.write('Hello, \x1b[1;31mWorld!\x1b[0m\r\n');
  renderer.write('\x1b[32mGreen text\x1b[0m on \x1b[44mblue background');

  // Render and save
  const pngBuffer = await renderer.renderToPNG();
  fs.writeFileSync('terminal.png', pngBuffer);
}

main().catch(console.error);