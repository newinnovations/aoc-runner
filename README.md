# AoC Runner – Live TUI runner for Advent of Code

A lightweight, terminal-based watcher and runner for Advent of Code (AoC) Python solutions. It automatically detects changes to your solution files or sample inputs, kills any running process, and immediately re-runs the appropriate script with the selected input. Output is displayed in a scrollable TUI with live timing and status information.

Perfect for rapid iteration during AoC – edit your code or samples and see results instantly.

## Features

- **Auto-reload on file changes** – Watches both parts (`aoc-[year]-[day][a/b].py`) and sample inputs (`ref.txt`, `ref2.txt` … `ref9.txt`).
- **Live output view** – Streaming stdout/stderr with automatic scrolling to bottom.
- **Multiple inputs** – Switch between puzzle input (`input.txt`) and up to 9 sample files.
- **Part A/B switching** – Toggle between part A and part B solutions.
- **Process control** – Kill a hanging run with `k`.
- **Performance timing** – Shows elapsed time while running and after completion.
- **Scrollable output** – Vertical and horizontal scrolling, page up/down, home/end.
- **Clean TUI** built with **ratatui** and **crossterm**.

## Directory Structure Expected

The program infers the year and day from your current working directory path:

```text
.../advent-of-code/2025/5
                   └─^──^── parsed as year=2025, day=5
```

Inside that directory you should have:

```text
aoc-2025-5a.py     # Part A solution
aoc-2025-5b.py     # Part B solution
input.txt          # Puzzle input (usually a symlink or copy)
ref.txt            # Sample input
ref2.txt           # Sample input 2 (optional)
...
ref9.txt           # Up to sample 9
```

## Usage

1. Place the compiled binary inside your path.

   ```bash
   cd path/to/advent-of-code/2025/5
   aoc
   ```

2. The program starts and immediately runs the current part with the current input.

3. Edit any watched file → the runner automatically restarts.

## Key Bindings

| Key                   | Action                                            |
| --------------------- | ------------------------------------------------- |
| `q`                   | Quit                                              |
| `a`                   | Switch to Part A                                  |
| `b`                   | Switch to Part B                                  |
| `i`                   | Use puzzle input (`input.txt`)                    |
| `1` – `9`             | Use sample input `ref.txt` or `refN.txt`          |
| `k`                   | Kill currently running process                    |
| `↑` / `↓`             | Scroll output vertically (hold Ctrl for faster)   |
| `←` / `→`             | Scroll output horizontally (hold Ctrl for faster) |
| `PageUp` / `PageDown` | Scroll by one page vertically                     |
| `Home`                | Jump to top-left of output                        |
| `End`                 | Jump to bottom of output                          |

## Building

```bash
cargo build --release
```

The binary will be at `target/release/aoc`

## Notes

- Output is capped at 1000 lines and 1000 characters per line to keep memory usage low.
- Errors from the Python script are prefixed with `ERROR:`.
- The runner assumes `python3` is available in `PATH`.
- `input.txt` is **not** watched for changes (common to keep it as a symlink).

Enjoy faster AoC sessions! 🎄
