const { HeadlessTerminal } = require('xterm-headless');
const { createCanvas, loadImage } = require('@napi-rs/canvas');

class TerminalRenderer {
  constructor(options = {}) {
    this.options = {
      cols: 80,
      rows: 24,
      fontSize: 14,
      fontFamily: 'monospace',
      ...options
    };

    this.terminal = new HeadlessTerminal({
      cols: this.options.cols,
      rows: this.options.rows
    });
    this.terminal.open();

    // Initialize font metrics
    this.charWidth = 0;
    this.charHeight = 0;
    this._initFontMetrics();
  }

  _initFontMetrics() {
    // Create temporary canvas for font metrics
    const canvas = createCanvas(100, 100);
    const ctx = canvas.getContext('2d');
    ctx.font = `${this.options.fontSize}px ${this.options.fontFamily}`;
    
    const metrics = ctx.measureText('W');
    this.charWidth = Math.ceil(metrics.width);
    this.charHeight = Math.ceil(
      metrics.actualBoundingBoxAscent + metrics.actualBoundingBoxDescent
    );
  }

  write(data) {
    this.terminal.write(data);
  }

  async renderToPNG() {
    const { cols, rows } = this.terminal;
    const canvas = createCanvas(
      cols * this.charWidth,
      rows * this.charHeight
    );
    const ctx = canvas.getContext('2d');
    ctx.font = `${this.options.fontSize}px ${this.options.fontFamily}`;
    ctx.textBaseline = 'top';

    const buffer = this.terminal.buffer.active;
    
    for (let y = 0; y < rows; y++) {
      const line = buffer.getLine(y);
      if (!line) continue;
      
      for (let x = 0; x < cols; x++) {
        const cell = line.getCell(x);
        if (!cell) continue;
        
        this._renderCell(ctx, x, y, cell);
      }
    }

    return canvas.encode('png');
  }

  _renderCell(ctx, x, y, cell) {
    const char = cell.getChars() || ' ';
    const bgColor = this._colorToCSS(cell.getBgColor(), true);
    const fgColor = this._colorToCSS(cell.getFgColor());

    // Draw background
    ctx.fillStyle = bgColor;
    ctx.fillRect(
      x * this.charWidth,
      y * this.charHeight,
      this.charWidth,
      this.charHeight
    );

    // Draw text
    ctx.fillStyle = fgColor;
    ctx.fillText(
      char,
      x * this.charWidth,
      y * this.charHeight
    );

    // Handle text styles
    if (cell.isBold()) ctx.globalAlpha = 0.7;
    if (cell.isUnderline()) {
      ctx.strokeStyle = fgColor;
      ctx.beginPath();
      ctx.moveTo(x * this.charWidth, (y + 1) * this.charHeight - 2);
      ctx.lineTo((x + 1) * this.charWidth, (y + 1) * this.charHeight - 2);
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0; // Reset
  }

  _colorToCSS(color, isBg = false) {
    // Default colors
    if (color === 0) return isBg ? '#000000' : '#ffffff';
    if (color === 1) return '#ff0000'; // Example: Red for errors

    // ANSI color lookup table (basic 16 colors)
    const ansiColors = [
      '#000000', '#aa0000', '#00aa00', '#aa5500',
      '#0000aa', '#aa00aa', '#00aaaa', '#aaaaaa',
      '#555555', '#ff5555', '#55ff55', '#ffff55',
      '#5555ff', '#ff55ff', '#55ffff', '#ffffff'
    ];

    // Extended colors (simplified)
    if (color >= 16 && color <= 231) {
      const idx = color - 16;
      const r = Math.floor(idx / 36) * 51;
      const g = Math.floor((idx % 36) / 6) * 51;
      const b = (idx % 6) * 51;
      return `rgb(${r},${g},${b})`;
    }

    // Grayscale
    if (color >= 232 && color <= 255) {
      const level = Math.floor((color - 232) * 10 + 8);
      return `rgb(${level},${level},${level})`;
    }

    return ansiColors[color] || (isBg ? '#000000' : '#ffffff');
  }
}

module.exports = TerminalRenderer;