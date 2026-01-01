use anyhow::{Context, Result};
use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState},
};
use std::{
    env,
    io::{self, BufRead, BufReader},
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::mpsc;

const MAX_LINES: usize = 1000;
const MAX_LINE_LENGTH: usize = 1000;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Part {
    A,
    B,
}

#[derive(Debug, Clone, PartialEq)]
enum Input {
    Main,
    Ref(u8),
}

#[derive(Debug)]
enum AppEvent {
    FileChanged,
    OutputLine(Color, String),
    ProcessFinished,
    Tick,
}

struct AppState {
    year: u32,
    day: u32,
    part: Part,
    input: Input,
    output_lines: Vec<(Color, String)>,
    scroll_offset: usize,
    horizontal_scroll: usize,
    viewport_height: usize,
    running: bool,
    start_time: Option<Instant>,
    end_time: Option<Instant>,
    child_process: Option<Child>,
}

impl AppState {
    fn new(year: u32, day: u32) -> Self {
        Self {
            year,
            day,
            part: Part::A,
            input: Input::Ref(1),
            output_lines: Vec::new(),
            scroll_offset: 0,
            horizontal_scroll: 0,
            viewport_height: 20,
            running: false,
            start_time: None,
            end_time: None,
            child_process: None,
        }
    }

    fn program_path(&self) -> PathBuf {
        PathBuf::from(format!(
            "aoc-{}-{}{}.py",
            self.year,
            self.day,
            match self.part {
                Part::A => 'a',
                Part::B => 'b',
            }
        ))
    }

    fn input_path(&self) -> PathBuf {
        PathBuf::from(match &self.input {
            Input::Main => "input.txt".to_string(),
            Input::Ref(n) => {
                if *n == 1 {
                    "ref.txt".to_string()
                } else {
                    format!("ref{}.txt", n)
                }
            }
        })
    }

    fn elapsed_time(&self) -> Option<Duration> {
        if let Some(start) = self.start_time {
            if let Some(end) = self.end_time {
                Some(end - start)
            } else if self.running {
                Some(start.elapsed())
            } else {
                None
            }
        } else {
            None
        }
    }

    fn add_output_line(&mut self, color: Color, line: String) {
        let truncated = if line.len() > MAX_LINE_LENGTH {
            line.chars().take(MAX_LINE_LENGTH).collect()
        } else {
            line
        };

        self.output_lines.push((color, truncated));
        if self.output_lines.len() > MAX_LINES {
            self.output_lines.remove(0);
            if self.scroll_offset > 0 {
                self.scroll_offset -= 1;
            }
        }
    }

    fn scroll_to_bottom(&mut self, viewport_height: usize) {
        self.scroll_offset = self.max_scroll_offset(viewport_height);
    }

    fn max_scroll_offset(&self, viewport_height: usize) -> usize {
        self.output_lines.len().saturating_sub(viewport_height)
    }

    fn kill_process(&mut self) {
        if let Some(mut child) = self.child_process.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        self.running = false;
        if self.start_time.is_some() && self.end_time.is_none() {
            self.end_time = Some(Instant::now());
        }
    }
}

fn parse_directory() -> Result<(u32, u32)> {
    let cwd = env::current_dir().context("Failed to get current directory")?;
    let path_str = cwd.to_string_lossy();

    let parts: Vec<&str> = path_str.split('/').collect();
    let len = parts.len();

    if len < 2 {
        anyhow::bail!("Directory structure doesn't match expected pattern");
    }

    let day: u32 = parts[len - 1]
        .parse()
        .context("Failed to parse day from directory")?;
    let year: u32 = parts[len - 2]
        .parse()
        .context("Failed to parse year from directory")?;

    Ok((year, day))
}

async fn watch_files(tx: mpsc::UnboundedSender<AppEvent>, year: u32, day: u32) -> Result<()> {
    let (file_tx, mut file_rx) = mpsc::unbounded_channel();

    let file_tx_clone = file_tx.clone();

    let mut watcher = RecommendedWatcher::new(
        move |res: Result<notify::Event, notify::Error>| {
            if let Ok(event) = res
                && matches!(
                    event.kind,
                    notify::EventKind::Modify(_) | notify::EventKind::Create(_)
                )
            {
                let _ = file_tx_clone.send(());
            }
        },
        Config::default(),
    )?;

    // Watch program files
    let prog_a = format!("aoc-{}-{}a.py", year, day);
    let prog_b = format!("aoc-{}-{}b.py", year, day);

    if Path::new(&prog_a).exists() {
        watcher.watch(Path::new(&prog_a), RecursiveMode::NonRecursive)?;
    }
    if Path::new(&prog_b).exists() {
        watcher.watch(Path::new(&prog_b), RecursiveMode::NonRecursive)?;
    }

    // Watch ref files (but not input.txt as it's a symlink)
    for i in 1..=9 {
        let ref_file = if i == 1 {
            "ref.txt".to_string()
        } else {
            format!("ref{}.txt", i)
        };
        if Path::new(&ref_file).exists() {
            watcher.watch(Path::new(&ref_file), RecursiveMode::NonRecursive)?;
        }
    }

    tokio::spawn(async move {
        while (file_rx.recv().await).is_some() {
            let _ = tx.send(AppEvent::FileChanged);
            // Debounce: wait a bit to avoid multiple rapid triggers
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        drop(watcher);
    });

    Ok(())
}

async fn run_program(
    state: Arc<Mutex<AppState>>,
    tx: mpsc::UnboundedSender<AppEvent>,
) -> Result<()> {
    let (prog_path, input_path) = {
        let state = state.lock().unwrap();
        (state.program_path(), state.input_path())
    };

    if !prog_path.exists() {
        tx.send(AppEvent::OutputLine(
            Color::LightRed,
            format!("Error: Program '{}' not found", prog_path.display()),
        ))?;
        tx.send(AppEvent::ProcessFinished)?;
        return Ok(());
    }

    if !input_path.exists() {
        tx.send(AppEvent::OutputLine(
            Color::LightRed,
            format!("Error: Input file '{}' not found", input_path.display()),
        ))?;
        tx.send(AppEvent::ProcessFinished)?;
        return Ok(());
    }

    let input_file = std::fs::File::open(&input_path).context("Failed to open input file")?;

    let mut child = Command::new("python3")
        .arg(&prog_path)
        .stdin(Stdio::from(input_file))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("Failed to spawn process")?;

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    {
        let mut state = state.lock().unwrap();
        state.child_process = Some(child);
    }

    let tx_stdout = tx.clone();
    let stdout_handle = tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            let _ = tx_stdout.send(AppEvent::OutputLine(Color::White, line));
        }
    });

    let tx_stderr = tx.clone();
    let stderr_handle = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            let _ = tx_stderr.send(AppEvent::OutputLine(
                Color::LightRed,
                format!("ERROR: {}", line),
            ));
        }
    });

    let state_clone = state.clone();
    tokio::spawn(async move {
        // Wait for stdout and stderr to finish reading
        let _ = tokio::join!(stdout_handle, stderr_handle);

        // Then wait for process to exit
        let result = {
            let mut state = state_clone.lock().unwrap();
            if let Some(child) = &mut state.child_process {
                child.wait()
            } else {
                return;
            }
        };

        if let Ok(status) = result {
            let msg = if status.success() {
                "✓ Process completed successfully".to_string()
            } else {
                format!(
                    "✗ Process exited with code: {}",
                    status
                        .code()
                        .map(|c| c.to_string())
                        .unwrap_or_else(|| "unknown".to_string())
                )
            };
            let _ = tx.send(AppEvent::OutputLine(Color::DarkGray, String::new()));
            let _ = tx.send(AppEvent::OutputLine(Color::DarkGray, msg));
        }
        let _ = tx.send(AppEvent::ProcessFinished);
    });

    Ok(())
}

fn render_ui(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut AppState,
) -> Result<()> {
    terminal.draw(|f| {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(3), Constraint::Length(3)])
            .split(f.area());

        // Output area
        let output_height = chunks[0].height.saturating_sub(2) as usize;
        state.viewport_height = output_height;

        let visible_start = state.scroll_offset;
        let visible_end = (visible_start + output_height).min(state.output_lines.len());

        // Apply horizontal scrolling
        let visible_lines: Vec<Line> = state.output_lines[visible_start..visible_end]
            .iter()
            .map(|(color, line)| {
                let chars: Vec<char> = line.chars().collect();
                let start = state.horizontal_scroll.min(chars.len());
                let substring: String = chars[start..].iter().collect();
                Line::from(Span::styled(substring, Style::default().fg(*color)))
            })
            .collect();

        let output_block = Block::default().borders(Borders::ALL).title(format!(
            " Output (lines {}-{} of {}{}) ",
            visible_start + 1,
            visible_end,
            state.output_lines.len(),
            if state.horizontal_scroll > 0 {
                format!(" | col {}", state.horizontal_scroll + 1)
            } else {
                String::new()
            }
        ));

        let output = Paragraph::new(visible_lines).block(output_block);
        f.render_widget(output, chunks[0]);

        // Scrollbar
        if state.output_lines.len() > output_height {
            let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
                .begin_symbol(Some("↑"))
                .end_symbol(Some("↓"));
            let max_scroll = state.max_scroll_offset(output_height);
            let mut scrollbar_state = ScrollbarState::new(max_scroll).position(state.scroll_offset);
            f.render_stateful_widget(
                scrollbar,
                chunks[0].inner(ratatui::layout::Margin {
                    vertical: 1,
                    horizontal: 0,
                }),
                &mut scrollbar_state,
            );
        }

        // Status bar
        let status_text = vec![
            Span::styled(
                format!("AoC {}/{} ", state.year, state.day),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw("| Part: "),
            Span::styled(
                match state.part {
                    Part::A => "A",
                    Part::B => "B",
                },
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" | Input: "),
            Span::styled(
                match &state.input {
                    Input::Main => "input.txt".to_string(),
                    Input::Ref(n) => {
                        if *n == 1 {
                            "ref.txt".to_string()
                        } else {
                            format!("ref{}.txt", n)
                        }
                    }
                },
                Style::default().fg(Color::Yellow),
            ),
            Span::raw(" | "),
            if state.running {
                Span::styled(
                    "RUNNING",
                    Style::default()
                        .fg(Color::LightRed)
                        .add_modifier(Modifier::BOLD),
                )
            } else {
                Span::styled("STOPPED", Style::default().fg(Color::Green))
            },
            Span::raw(" | Time: "),
            Span::styled(
                if let Some(duration) = state.elapsed_time() {
                    format!("{:.3}s", duration.as_secs_f64())
                } else {
                    "—".to_string()
                },
                Style::default().fg(Color::Magenta),
            ),
            Span::raw(" | "),
            Span::styled(
                "[a/b] Part, [1-9] Sample data, [i] Puzzle data, [k] Kill, [q] Quit",
                Style::default().fg(Color::DarkGray),
            ),
        ];

        let status = Paragraph::new(Line::from(status_text))
            .block(Block::default().borders(Borders::ALL).title(" Status "));
        f.render_widget(status, chunks[1]);
    })?;

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let (year, day) = parse_directory().context("Failed to parse directory structure")?;

    let state = Arc::new(Mutex::new(AppState::new(year, day)));
    let (tx, mut rx) = mpsc::unbounded_channel();

    // Start file watcher
    watch_files(tx.clone(), year, day).await?;

    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Trigger initial run
    let _ = tx.send(AppEvent::FileChanged);

    // Tick timer for UI updates when running
    let tx_tick = tx.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(100)).await;
            let _ = tx_tick.send(AppEvent::Tick);
        }
    });

    // Main loop
    loop {
        {
            let mut state_guard = state.lock().unwrap();
            render_ui(&mut terminal, &mut state_guard)?;
        }

        // Handle events
        if event::poll(Duration::from_millis(50))?
            && let Event::Key(key) = event::read()?
        {
            let mut state_guard = state.lock().unwrap();

            let delta = if key.modifiers.contains(KeyModifiers::CONTROL) {
                50
            } else {
                1
            };

            match key.code {
                KeyCode::Char('q') => break,
                KeyCode::Char('a') => {
                    if state_guard.part != Part::A {
                        state_guard.part = Part::A;
                        drop(state_guard);
                        let _ = tx.send(AppEvent::FileChanged);
                    }
                }
                KeyCode::Char('b') => {
                    if state_guard.part != Part::B {
                        state_guard.part = Part::B;
                        drop(state_guard);
                        let _ = tx.send(AppEvent::FileChanged);
                    }
                }
                KeyCode::Char('i') => {
                    if state_guard.input != Input::Main {
                        state_guard.input = Input::Main;
                        drop(state_guard);
                        let _ = tx.send(AppEvent::FileChanged);
                    }
                }
                KeyCode::Char(c @ '1'..='9') => {
                    let n = c.to_digit(10).unwrap() as u8;
                    if state_guard.input != Input::Ref(n) {
                        state_guard.input = Input::Ref(n);
                        drop(state_guard);
                        let _ = tx.send(AppEvent::FileChanged);
                    }
                }
                KeyCode::Char('k') => {
                    state_guard.kill_process();
                }
                KeyCode::Up => {
                    state_guard.scroll_offset = state_guard.scroll_offset.saturating_sub(delta);
                }
                KeyCode::Down => {
                    let viewport_height = state_guard.viewport_height;
                    let max = state_guard.max_scroll_offset(viewport_height);
                    state_guard.scroll_offset = (state_guard.scroll_offset + delta).min(max);
                }
                KeyCode::PageUp => {
                    let viewport_height = state_guard.viewport_height;
                    state_guard.scroll_offset =
                        state_guard.scroll_offset.saturating_sub(viewport_height);
                }
                KeyCode::PageDown => {
                    let viewport_height = state_guard.viewport_height;
                    let max = state_guard.max_scroll_offset(viewport_height);
                    state_guard.scroll_offset =
                        (state_guard.scroll_offset + viewport_height).min(max);
                }
                KeyCode::Left => {
                    state_guard.horizontal_scroll =
                        state_guard.horizontal_scroll.saturating_sub(delta);
                }
                KeyCode::Right => {
                    state_guard.horizontal_scroll =
                        state_guard.horizontal_scroll.saturating_add(delta);
                }
                KeyCode::Home => {
                    state_guard.scroll_offset = 0;
                    state_guard.horizontal_scroll = 0;
                }
                KeyCode::End => {
                    let viewport_height = state_guard.viewport_height;
                    state_guard.scroll_to_bottom(viewport_height);
                }
                _ => {}
            }
        }

        // Process async events
        while let Ok(event) = rx.try_recv() {
            match event {
                AppEvent::FileChanged => {
                    let mut state_guard = state.lock().unwrap();
                    state_guard.kill_process();
                    state_guard.output_lines.clear();
                    state_guard.scroll_offset = 0;
                    state_guard.running = true;
                    state_guard.start_time = Some(Instant::now());
                    state_guard.end_time = None;
                    drop(state_guard);

                    let state_clone = state.clone();
                    let tx_clone = tx.clone();
                    tokio::spawn(async move {
                        let _ = run_program(state_clone, tx_clone).await;
                    });
                }
                AppEvent::OutputLine(color, line) => {
                    let mut state_guard = state.lock().unwrap();
                    state_guard.add_output_line(color, line);
                    let viewport_height = state_guard.viewport_height;
                    state_guard.scroll_to_bottom(viewport_height);
                }
                AppEvent::ProcessFinished => {
                    let mut state_guard = state.lock().unwrap();
                    state_guard.running = false;
                    state_guard.end_time = Some(Instant::now());
                    state_guard.child_process = None;
                }
                AppEvent::Tick => {
                    // Just triggers a redraw
                }
            }
        }
    }

    // Cleanup
    {
        let mut state_guard = state.lock().unwrap();
        state_guard.kill_process();
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;

    Ok(())
}
