//! A debug overlay drawn into each presented frame. Alt+Z, while our
//! window has focus, shows and hides it (`VICTORY_OVERLAY=0` turns it off,
//! `VICTORY_OVERLAY=1` shows it from the start).
//!
//! It shows the frame rate and frame times (with a graph of the last
//! frames and the 1% low), the presenter's last second (draws per frame,
//! time preparing submissions and waiting for the GPU), the process's CPU
//! use and memory, video memory against its budget, and the controller as
//! the title last read it. Keys 1-4, while it shows, leave kinds of GPU
//! work out ([`skip`]), to see what draws what. While it shows the title
//! gets no keyboard or controller input ([`captured`]).
//!
//! The panel is drawn on the CPU (our own 5x7 font) into upload memory in
//! the back buffer's format, and copied into its top left corner after the
//! frame, before the present: no shaders or pipelines, and nothing at all
//! while it is hidden.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::HWND;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;

use crate::host::Host;

static VISIBLE: AtomicBool = AtomicBool::new(false);

/// Whether the overlay shows: the title gets no input meanwhile.
pub fn captured() -> bool {
    VISIBLE.load(Relaxed)
}

/// Kinds of GPU work the toggles can leave out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Work {
    Compute = 0,
    Mesh = 1,
    Vertex = 2,
    Indirect = 3,
}

static SKIPPED: [AtomicBool; 4] = [const { AtomicBool::new(false) }; 4];
const WORK: [&str; 4] = [
    "COMPUTE DISPATCHES",
    "MESH DRAWS",
    "VERTEX DRAWS",
    "INDIRECT EXECUTION",
];

/// Whether the toggles leave `work` out (one relaxed load).
#[inline]
pub(crate) fn skip(work: Work) -> bool {
    SKIPPED[work as usize].load(Relaxed)
}

/// A controller as the title reads it: GameInput's buttons, triggers and
/// sticks (-1 to 1, y up).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Pad {
    pub buttons: u32,
    pub triggers: [f32; 2],
    pub left: [f32; 2],
    pub right: [f32; 2],
}

static PAD: Mutex<Option<Pad>> = Mutex::new(None);

/// The controller the title reads now (`None`: none).
pub fn show_pad(pad: Option<Pad>) {
    *PAD.lock().unwrap_or_else(|e| e.into_inner()) = pad;
}

/// The presenter's last second, per frame (`crate::host::Meter`).
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Second {
    pub(crate) draws: f64,
    pub(crate) prepare_ms: f64,
    pub(crate) gpu_wait_ms: f64,
}

static SECOND: Mutex<Second> = Mutex::new(Second {
    draws: 0.0,
    prepare_ms: 0.0,
    gpu_wait_ms: 0.0,
});

pub(crate) fn second(s: Second) {
    *SECOND.lock().unwrap_or_else(|e| e.into_inner()) = s;
}

/// Frame times kept for the graph and the 1% low.
const HISTORY: usize = 240;
/// The panel is redrawn at most this often; it is copied every frame.
const REDRAW: Duration = Duration::from_millis(250);
/// The process and video memory are asked at most this often.
const SAMPLE: Duration = Duration::from_secs(1);
/// The panel's distance from the corner, in font pixels.
const MARGIN: u32 = 6;

/// The presenter's overlay state.
pub(crate) struct Overlay {
    enabled: bool,
    /// Keys down at the last frame: Alt+Z, then 1-4.
    down: u8,
    last: Option<Instant>,
    times: VecDeque<f32>,
    sampled: Option<(Instant, System)>,
    cpu: Option<(Instant, u64)>,
    adapter: Option<Option<IDXGIAdapter3>>,
    /// Upload memory (its address as a number: the presenter lives behind
    /// a lock other threads take), its size, and when it was last drawn.
    upload: Option<(ID3D12Resource, usize, u64)>,
    drawn: Option<(Instant, Image, i32)>,
}

impl Default for Overlay {
    fn default() -> Self {
        let switch = std::env::var_os("VICTORY_OVERLAY");
        if switch.as_deref().is_some_and(|v| v == "1") {
            VISIBLE.store(true, Relaxed);
        }
        Overlay {
            enabled: switch.as_deref().is_none_or(|v| v != "0"),
            down: 0,
            last: None,
            times: VecDeque::with_capacity(HISTORY),
            sampled: None,
            cpu: None,
            adapter: None,
            upload: None,
            drawn: None,
        }
    }
}

/// What is sampled once a second.
#[derive(Clone, Copy, Debug, Default)]
struct System {
    cpu: Option<f64>,
    private: u64,
    working: u64,
    ram: (u64, u64),
    vram: Option<(u64, u64, u64)>,
}

impl Overlay {
    /// Counts the frame, polls the keys and, if the panel shows and fits,
    /// records its copy into `back` (`width` x `height` of `format`, in
    /// `state`) on `list`, moving `back` to the copy destination state
    /// first. Returns the state `back` is left in.
    pub(crate) fn record(
        &mut self,
        host: &Host,
        list: &ID3D12GraphicsCommandList,
        back: &ID3D12Resource,
        (width, height, format): (u32, u32, DXGI_FORMAT),
        hwnd: HWND,
        state: D3D12_RESOURCE_STATES,
    ) -> D3D12_RESOURCE_STATES {
        if !self.enabled {
            return state;
        }
        let now = Instant::now();
        if let Some(last) = self.last.replace(now) {
            if self.times.len() == HISTORY {
                self.times.pop_front();
            }
            self.times.push_back((now - last).as_secs_f32() * 1000.0);
        }
        self.poll_keys(hwnd);
        if !captured() {
            self.drawn = None;
            return state;
        }
        let Some(pixel) = Pixel::of(format) else {
            return state;
        };
        if self
            .sampled
            .as_ref()
            .is_none_or(|(at, _)| now - *at >= SAMPLE)
        {
            let system = self.sample(host);
            self.sampled = Some((now, system));
        }
        let scale = (height / 720).clamp(1, 3);
        let fresh = self.drawn.as_ref().is_some_and(|(at, image, f)| {
            now - *at < REDRAW && image.scale == scale && *f == format.0
        });
        if !fresh {
            let system = self.sampled.map(|(_, s)| s).unwrap_or_default();
            let image = self.panel(scale, &system);
            self.drawn = Some((now, image, -1));
        }
        let Some((_, image, written)) = self.drawn.as_mut() else {
            return state;
        };
        let margin = MARGIN * image.scale;
        if image.width + margin > width || image.height + margin > height {
            return state;
        }
        let row_pitch =
            (image.width * pixel.bytes()).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
        let size = u64::from(row_pitch) * u64::from(image.height);
        if self.upload.as_ref().is_none_or(|&(_, _, have)| have < size) {
            *written = -1;
            self.upload = host
                .upload_buffer(size)
                .map(|(buffer, data)| (buffer, data as usize, size));
        }
        let Some((buffer, data, have)) = &self.upload else {
            return state;
        };
        if *written != format.0 {
            // SAFETY: the buffer is mapped for `have` bytes, and the GPU is
            // done with it: every present's submission is waited for.
            let out = unsafe { std::slice::from_raw_parts_mut(*data as *mut u8, *have as usize) };
            image.write(pixel, row_pitch as usize, out);
            *written = format.0;
        }
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: a borrowed pointer the location does not release.
            pResource: unsafe { std::mem::transmute_copy(back) },
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let src = D3D12_TEXTURE_COPY_LOCATION {
            // SAFETY: as above.
            pResource: unsafe { std::mem::transmute_copy(buffer) },
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: format,
                        Width: image.width,
                        Height: image.height,
                        Depth: 1,
                        RowPitch: row_pitch,
                    },
                },
            },
        };
        // SAFETY: an open list; the back buffer is made a copy destination
        // and the upload buffer lives in `self` past the submission.
        unsafe {
            if state != D3D12_RESOURCE_STATE_COPY_DEST {
                list.ResourceBarrier(&[crate::host::transition(
                    back,
                    state,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                )]);
            }
            list.CopyTextureRegion(&dst, margin, margin, 0, &src, None);
        }
        D3D12_RESOURCE_STATE_COPY_DEST
    }

    /// Alt+Z shows and hides the panel; 1-4 toggle work while it shows.
    /// Only while our window has focus, on the frame a key goes down.
    fn poll_keys(&mut self, hwnd: HWND) {
        use windows_sys::Win32::UI::Input::KeyboardAndMouse::GetAsyncKeyState;
        // SAFETY: plain queries.
        let focused = unsafe { GetForegroundWindow() } == hwnd;
        // SAFETY: a plain query.
        let held = |vk: i32| focused && unsafe { GetAsyncKeyState(vk) } < 0;
        const ALT: i32 = 0x12;
        let mut down = u8::from(held(ALT) && held(i32::from(b'Z')));
        for (k, key) in (b'1'..=b'4').enumerate() {
            down |= u8::from(held(i32::from(key))) << (k + 1);
        }
        let pressed = down & !self.down;
        self.down = down;
        if pressed & 1 != 0 {
            let shown = !VISIBLE.fetch_xor(true, Relaxed);
            tracing::info!(shown, "overlay");
        }
        if captured() {
            for (k, skipped) in SKIPPED.iter().enumerate() {
                if pressed >> (k + 1) & 1 != 0 {
                    let off = !skipped.fetch_xor(true, Relaxed);
                    tracing::warn!(work = WORK[k], left_out = off, "overlay toggle");
                    self.drawn = None;
                }
            }
        }
    }

    /// The process's CPU use and memory, the machine's memory and the
    /// device's video memory.
    fn sample(&mut self, host: &Host) -> System {
        use windows_sys::Win32::System::ProcessStatus::{
            K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX,
        };
        use windows_sys::Win32::System::SystemInformation::{
            GetSystemInfo, GlobalMemoryStatusEx, MEMORYSTATUSEX, SYSTEM_INFO,
        };
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
        let mut s = System::default();
        // SAFETY: plain queries into structs of ours.
        unsafe {
            let (mut created, mut exited, mut kernel, mut user) = std::mem::zeroed();
            if GetProcessTimes(
                GetCurrentProcess(),
                &mut created,
                &mut exited,
                &mut kernel,
                &mut user,
            ) != 0
            {
                let ticks = |f: windows_sys::Win32::Foundation::FILETIME| {
                    u64::from(f.dwHighDateTime) << 32 | u64::from(f.dwLowDateTime)
                };
                let busy = ticks(kernel) + ticks(user);
                let now = Instant::now();
                if let Some((at, before)) = self.cpu.replace((now, busy)) {
                    let mut info: SYSTEM_INFO = std::mem::zeroed();
                    GetSystemInfo(&mut info);
                    let wall = (now - at).as_nanos() as f64 / 100.0;
                    let cores = f64::from(info.dwNumberOfProcessors.max(1));
                    s.cpu = Some(busy.saturating_sub(before) as f64 / wall / cores * 100.0);
                }
            }
            let mut m: PROCESS_MEMORY_COUNTERS_EX = std::mem::zeroed();
            m.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
            if K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut m).cast(), m.cb) != 0 {
                (s.private, s.working) = (m.PrivateUsage as u64, m.WorkingSetSize as u64);
            }
            let mut g: MEMORYSTATUSEX = std::mem::zeroed();
            g.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
            if GlobalMemoryStatusEx(&mut g) != 0 {
                s.ram = (g.ullTotalPhys - g.ullAvailPhys, g.ullTotalPhys);
            }
        }
        let adapter = self.adapter.get_or_insert_with(|| {
            // SAFETY: plain DXGI queries.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                factory
                    .EnumAdapterByLuid::<IDXGIAdapter3>(host.device.GetAdapterLuid())
                    .ok()
            }
        });
        if let Some(adapter) = adapter {
            let query = |group| {
                let mut info = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
                // SAFETY: a live adapter and a struct of ours.
                unsafe { adapter.QueryVideoMemoryInfo(0, group, &mut info) }
                    .ok()
                    .map(|()| info)
            };
            if let (Some(local), Some(shared)) = (
                query(DXGI_MEMORY_SEGMENT_GROUP_LOCAL),
                query(DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL),
            ) {
                s.vram = Some((local.CurrentUsage, local.Budget, shared.CurrentUsage));
            }
        }
        s
    }

    /// The panel's text and graph.
    fn panel(&self, scale: u32, system: &System) -> Image {
        let second = *SECOND.lock().unwrap_or_else(|e| e.into_inner());
        let pad = *PAD.lock().unwrap_or_else(|e| e.into_inner());
        let lines = lines(&self.times, &second, system, pad.as_ref());
        Image::panel(scale, &lines, &self.times)
    }
}

/// A panel line: its text and colour.
type Line = (String, Colour);

/// Gigabytes, one decimal.
fn gb(bytes: u64) -> String {
    format!("{:.1}", bytes as f64 / (1u64 << 30) as f64)
}

/// The panel's lines (the graph goes after the second).
fn lines(times: &VecDeque<f32>, second: &Second, s: &System, pad: Option<&Pad>) -> Vec<Line> {
    let mut out = vec![("VICTORY OVERLAY   ALT+Z HIDES".to_owned(), Colour::Dim)];
    let n = times.len().max(1) as f32;
    let mean = times.iter().sum::<f32>() / n;
    let mut sorted: Vec<f32> = times.iter().copied().collect();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let worst = sorted.iter().take((sorted.len() / 100).max(1)).sum::<f32>()
        / (sorted.len() / 100).max(1) as f32;
    let fps = |ms: f32| if ms > 0.0 { 1000.0 / ms } else { 0.0 };
    let rate = fps(mean);
    let colour = if rate >= 29.0 {
        Colour::Good
    } else if rate >= 20.0 {
        Colour::Warn
    } else {
        Colour::Bad
    };
    out.push((
        format!(
            "FPS {:5.1}  {:6.1} MS   1% LOW {:5.1}",
            rate,
            mean,
            fps(worst)
        ),
        colour,
    ));
    out.push((
        format!(
            "DRAWS {:5.0}  PREP {:5.1}MS  GPU WAIT {:5.1}MS",
            second.draws, second.prepare_ms, second.gpu_wait_ms
        ),
        Colour::Text,
    ));
    out.push((
        match s.cpu {
            Some(cpu) => format!(
                "CPU {cpu:5.1}%   RAM {} GB PRIVATE  {} GB IN USE",
                gb(s.private),
                gb(s.working)
            ),
            None => format!(
                "CPU   ...    RAM {} GB PRIVATE  {} GB IN USE",
                gb(s.private),
                gb(s.working)
            ),
        },
        Colour::Text,
    ));
    out.push((
        format!("SYSTEM RAM {} / {} GB", gb(s.ram.0), gb(s.ram.1)),
        Colour::Text,
    ));
    out.push((
        match s.vram {
            Some((used, budget, shared)) => {
                format!(
                    "VRAM {} / {} GB   SHARED {} GB",
                    gb(used),
                    gb(budget),
                    gb(shared)
                )
            }
            None => "VRAM ...".to_owned(),
        },
        Colour::Text,
    ));
    match pad {
        Some(p) => {
            out.push((
                format!(
                    "PAD L {:+.2} {:+.2}  R {:+.2} {:+.2}  LT {:.2} RT {:.2}",
                    p.left[0], p.left[1], p.right[0], p.right[1], p.triggers[0], p.triggers[1]
                ),
                Colour::Text,
            ));
            const NAMES: [&str; 14] = [
                "MENU", "VIEW", "A", "B", "X", "Y", "DU", "DD", "DL", "DR", "LB", "RB", "LS", "RS",
            ];
            let held: Vec<&str> = NAMES
                .iter()
                .enumerate()
                .filter(|&(k, _)| p.buttons >> k & 1 != 0)
                .map(|(_, n)| *n)
                .collect();
            out.push((
                format!(
                    "    {}",
                    if held.is_empty() {
                        "-".to_owned()
                    } else {
                        held.join(" ")
                    }
                ),
                Colour::Text,
            ));
        }
        None => out.push(("PAD NONE".to_owned(), Colour::Dim)),
    }
    out.push(("KEYS 1-4 LEAVE WORK OUT:".to_owned(), Colour::Dim));
    for (k, name) in WORK.iter().enumerate() {
        let off = SKIPPED[k].load(Relaxed);
        out.push((
            format!(" {} [{}] {}", k + 1, if off { " " } else { "X" }, name),
            if off { Colour::Bad } else { Colour::Text },
        ));
    }
    out
}

/// Panel colours.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Colour {
    Back = 0,
    Text = 1,
    Dim = 2,
    Good = 3,
    Warn = 4,
    Bad = 5,
}

/// Each colour's linear RGB (the float formats) and sRGB-encoded (the
/// others: the back buffer's bytes are shown as they are).
const RGB: [[f32; 3]; 6] = [
    [0.02, 0.02, 0.03],
    [0.92, 0.92, 0.92],
    [0.55, 0.55, 0.60],
    [0.35, 0.90, 0.40],
    [0.95, 0.80, 0.25],
    [0.95, 0.35, 0.30],
];

/// The back buffer's pixel formats the presenter makes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pixel {
    Rgba8,
    Bgra8,
    Rgb10a2,
    Rgba16f,
}

impl Pixel {
    fn of(format: DXGI_FORMAT) -> Option<Pixel> {
        Some(match format {
            DXGI_FORMAT_R8G8B8A8_UNORM => Pixel::Rgba8,
            DXGI_FORMAT_B8G8R8A8_UNORM => Pixel::Bgra8,
            DXGI_FORMAT_R10G10B10A2_UNORM => Pixel::Rgb10a2,
            DXGI_FORMAT_R16G16B16A16_FLOAT => Pixel::Rgba16f,
            _ => return None,
        })
    }

    fn bytes(self) -> u32 {
        if self == Pixel::Rgba16f { 8 } else { 4 }
    }

    /// Colour `c` in this format.
    fn encode(self, c: Colour) -> [u8; 8] {
        let [r, g, b] = RGB[c as usize];
        let unorm = |x: f32, max: f32| (x.clamp(0.0, 1.0) * max).round() as u32;
        let mut out = [0u8; 8];
        match self {
            Pixel::Rgba8 => {
                out[..4].copy_from_slice(&[r, g, b, 1.0].map(|x| unorm(x, 255.0) as u8))
            }
            Pixel::Bgra8 => {
                out[..4].copy_from_slice(&[b, g, r, 1.0].map(|x| unorm(x, 255.0) as u8))
            }
            Pixel::Rgb10a2 => {
                let v =
                    unorm(r, 1023.0) | unorm(g, 1023.0) << 10 | unorm(b, 1023.0) << 20 | 3 << 30;
                out[..4].copy_from_slice(&v.to_le_bytes());
            }
            Pixel::Rgba16f => {
                // The float formats are linear: decode from sRGB.
                let linear = |x: f32| x.powf(2.2);
                for (k, x) in [linear(r), linear(g), linear(b), 1.0]
                    .into_iter()
                    .enumerate()
                {
                    out[2 * k..2 * k + 2].copy_from_slice(&half(x).to_le_bytes());
                }
            }
        }
        out
    }
}

/// `x` (0 to 1) as a half float.
fn half(x: f32) -> u16 {
    if x <= 0.0 {
        return 0;
    }
    let bits = x.to_bits();
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    if exp <= 0 {
        return 0;
    }
    ((exp as u32) << 10 | (bits >> 13 & 0x3FF)) as u16
}

/// The panel in colours, `scale` pixels per font pixel.
#[derive(Clone, Debug, PartialEq)]
struct Image {
    width: u32,
    height: u32,
    scale: u32,
    pixels: Vec<Colour>,
}

/// A character cell: 5x7 glyphs, a column and two rows apart.
const CELL: (u32, u32) = (6, 9);
/// Characters per line.
const COLUMNS: u32 = 50;
/// The graph's height, in font pixels.
const GRAPH: u32 = 30;
/// Frame times at and past this fill the graph.
const GRAPH_MS: f32 = 66.7;

impl Image {
    fn panel(scale: u32, lines: &[Line], times: &VecDeque<f32>) -> Image {
        let (cw, ch) = CELL;
        let width = (COLUMNS * cw + 4) * scale;
        let rows = lines.len() as u32;
        let height = (rows * ch + GRAPH + 8) * scale;
        let mut image = Image {
            width,
            height,
            scale,
            pixels: vec![Colour::Back; (width * height) as usize],
        };
        let mut y = 2;
        for (k, (text, colour)) in lines.iter().enumerate() {
            image.text(2, y, text, *colour);
            y += ch;
            // The graph after the frame-rate line.
            if k == 1 {
                image.graph(2, y + 1, COLUMNS * cw, GRAPH - 2, times);
                y += GRAPH;
            }
        }
        image
    }

    /// Fills font pixel (`x`, `y`).
    fn dot(&mut self, x: u32, y: u32, c: Colour) {
        for dy in 0..self.scale {
            for dx in 0..self.scale {
                let (px, py) = (x * self.scale + dx, y * self.scale + dy);
                if px < self.width && py < self.height {
                    self.pixels[(py * self.width + px) as usize] = c;
                }
            }
        }
    }

    fn text(&mut self, x: u32, y: u32, text: &str, c: Colour) {
        for (k, ch) in text.chars().take(COLUMNS as usize).enumerate() {
            let rows = glyph(ch.to_ascii_uppercase());
            let left = x + k as u32 * CELL.0;
            for (row, bits) in rows.iter().enumerate() {
                for col in 0..5 {
                    if bits >> (4 - col) & 1 != 0 {
                        self.dot(left + col, y + row as u32, c);
                    }
                }
            }
        }
    }

    /// Frame times as bars, newest on the right, one font pixel each.
    fn graph(&mut self, x: u32, y: u32, width: u32, height: u32, times: &VecDeque<f32>) {
        for k in 0..width {
            self.dot(x + k, y + height, Colour::Dim);
        }
        // The 33 ms line (30 fps).
        let at_30 = (33.3 / GRAPH_MS * height as f32) as u32;
        for k in (0..width).step_by(3) {
            self.dot(x + k, y + height - at_30.min(height), Colour::Dim);
        }
        let shown = times.len().min(width as usize);
        for (k, &ms) in times.iter().skip(times.len() - shown).enumerate() {
            let bar = ((ms / GRAPH_MS).min(1.0) * height as f32).round() as u32;
            let colour = if ms <= 34.0 {
                Colour::Good
            } else if ms <= 50.0 {
                Colour::Warn
            } else {
                Colour::Bad
            };
            let column = x + width - shown as u32 + k as u32;
            for row in 0..bar {
                self.dot(column, y + height - 1 - row, colour);
            }
        }
    }

    /// The image's rows in `pixel`'s format, `pitch` bytes apart.
    fn write(&self, pixel: Pixel, pitch: usize, out: &mut [u8]) {
        let palette: Vec<[u8; 8]> = (0..6)
            .map(|k| {
                pixel.encode(match k {
                    0 => Colour::Back,
                    1 => Colour::Text,
                    2 => Colour::Dim,
                    3 => Colour::Good,
                    4 => Colour::Warn,
                    _ => Colour::Bad,
                })
            })
            .collect();
        let n = pixel.bytes() as usize;
        for (y, row) in self.pixels.chunks(self.width as usize).enumerate() {
            let line = &mut out[y * pitch..y * pitch + row.len() * n];
            for (px, c) in line.chunks_mut(n).zip(row) {
                px.copy_from_slice(&palette[*c as usize][..n]);
            }
        }
    }
}

/// A character's 5x7 glyph, a row per byte (bit 4 leftmost); `?` for
/// those it has none of.
fn glyph(c: char) -> [u8; 7] {
    match c {
        ' ' => [0; 7],
        '0' => [0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E],
        '1' => [0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E],
        '2' => [0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F],
        '3' => [0x1F, 0x02, 0x04, 0x02, 0x01, 0x11, 0x0E],
        '4' => [0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02],
        '5' => [0x1F, 0x10, 0x1E, 0x01, 0x01, 0x11, 0x0E],
        '6' => [0x06, 0x08, 0x10, 0x1E, 0x11, 0x11, 0x0E],
        '7' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08],
        '8' => [0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E],
        '9' => [0x0E, 0x11, 0x11, 0x0F, 0x01, 0x02, 0x0C],
        'A' => [0x0E, 0x11, 0x11, 0x11, 0x1F, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1C, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1C],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x11, 0x0A, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '.' => [0, 0, 0, 0, 0, 0x0C, 0x0C],
        ',' => [0, 0, 0, 0, 0x0C, 0x04, 0x08],
        ':' => [0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '/' => [0, 0x01, 0x02, 0x04, 0x08, 0x10, 0],
        '-' => [0, 0, 0, 0x1F, 0, 0, 0],
        '+' => [0, 0x04, 0x04, 0x1F, 0x04, 0x04, 0],
        '(' => [0x02, 0x04, 0x08, 0x08, 0x08, 0x04, 0x02],
        ')' => [0x08, 0x04, 0x02, 0x02, 0x02, 0x04, 0x08],
        '[' => [0x0E, 0x08, 0x08, 0x08, 0x08, 0x08, 0x0E],
        ']' => [0x0E, 0x02, 0x02, 0x02, 0x02, 0x02, 0x0E],
        '=' => [0, 0, 0x1F, 0, 0x1F, 0, 0],
        '<' => [0x02, 0x04, 0x08, 0x10, 0x08, 0x04, 0x02],
        '>' => [0x08, 0x04, 0x02, 0x01, 0x02, 0x04, 0x08],
        '!' => [0x04, 0x04, 0x04, 0x04, 0, 0, 0x04],
        '_' => [0, 0, 0, 0, 0, 0, 0x1F],
        '|' => [0x04; 7],
        '*' => [0, 0x04, 0x15, 0x0E, 0x15, 0x04, 0],
        _ => [0x0E, 0x11, 0x01, 0x02, 0x04, 0, 0x04],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every character the panel prints has a glyph of its own (not the
    /// `?` stand-in), and the lines fit the panel.
    #[test]
    fn the_panel_prints_with_its_own_glyphs_and_fits() {
        let times: VecDeque<f32> = (0..HISTORY).map(|k| 20.0 + (k % 50) as f32).collect();
        let second = Second {
            draws: 2513.0,
            prepare_ms: 12.3,
            gpu_wait_ms: 18.2,
        };
        let system = System {
            cpu: Some(37.5),
            private: 14 << 30,
            working: 11 << 30,
            ram: (20 << 30, 64 << 30),
            vram: Some((8 << 30, 15 << 30, 1 << 29)),
        };
        let pad = Pad {
            buttons: 0x3FFF,
            triggers: [0.5, 1.0],
            left: [-1.0, 0.25],
            right: [0.0, -0.5],
        };
        let lines = lines(&times, &second, &system, Some(&pad));
        let unknown = glyph('\u{1}');
        for (text, _) in &lines {
            assert!(text.len() <= COLUMNS as usize, "{text}");
            for c in text.chars() {
                assert!(
                    c == '?' || glyph(c.to_ascii_uppercase()) != unknown,
                    "no glyph for {c:?} in {text}"
                );
            }
        }
        let image = Image::panel(2, &lines, &times);
        assert_eq!(image.pixels.len(), (image.width * image.height) as usize);
        assert!(image.pixels.contains(&Colour::Text) && image.pixels.contains(&Colour::Good));
    }

    /// The 1% low is the slowest 1% of frames; the rate colours follow it.
    #[test]
    fn frame_rate_line_takes_the_mean_and_the_slowest_frames() {
        let mut times: VecDeque<f32> = std::iter::repeat_n(33.3, 199).collect();
        times.push_back(100.0);
        times.push_back(100.0);
        let lines = lines(&times, &Second::default(), &System::default(), None);
        assert!(lines[1].0.contains("1% LOW  10.0"), "{}", lines[1].0);
        assert!(lines.iter().any(|(t, _)| t == "PAD NONE"));
    }

    /// The panel, shown, lands in the back buffer's top left corner on the
    /// host GPU: its background at the margin, its text in the text colour,
    /// the frame around it untouched.
    #[test]
    fn a_shown_panel_is_copied_into_the_back_buffer() {
        let Some(host) = crate::host::host() else {
            return; // no host GPU here
        };
        let (width, height) = (640u32, 400u32);
        let make = |heap: D3D12_HEAP_TYPE, desc: &D3D12_RESOURCE_DESC, state| {
            let mut r: Option<ID3D12Resource> = None;
            // SAFETY: plain resource creation.
            unsafe {
                host.device
                    .CreateCommittedResource(
                        &D3D12_HEAP_PROPERTIES {
                            Type: heap,
                            ..Default::default()
                        },
                        D3D12_HEAP_FLAG_NONE,
                        desc,
                        state,
                        None,
                        &mut r,
                    )
                    .unwrap();
            }
            r.unwrap()
        };
        let back = make(
            D3D12_HEAP_TYPE_DEFAULT,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: u64::from(width),
                Height: height,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let pitch = width * 4;
        let readback = make(
            D3D12_HEAP_TYPE_READBACK,
            &D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: u64::from(pitch * height),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            },
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let list = host.new_list().unwrap();
        let mut overlay = Overlay {
            enabled: true,
            ..Overlay::default()
        };
        VISIBLE.store(true, Relaxed);
        overlay.record(
            host,
            &list.list,
            &back,
            (width, height, DXGI_FORMAT_R8G8B8A8_UNORM),
            HWND::default(),
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        VISIBLE.store(false, Relaxed);
        // SAFETY: an open list and live resources; footprint in the buffer.
        unsafe {
            list.list.ResourceBarrier(&[crate::host::transition(
                &back,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                            Width: width,
                            Height: height,
                            Depth: 1,
                            RowPitch: pitch,
                        },
                    },
                },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&back),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        }
        list.close();
        host.execute(&[&list]);
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the copy finished (execute waits); the buffer holds the rows.
        let pixels = unsafe {
            readback.Map(0, None, Some(&mut mapped)).unwrap();
            std::slice::from_raw_parts(mapped.cast::<[u8; 4]>(), (width * height) as usize).to_vec()
        };
        let at = |x: u32, y: u32| pixels[(y * width + x) as usize];
        let back_colour: [u8; 4] = Pixel::Rgba8.encode(Colour::Back)[..4].try_into().unwrap();
        let text: [u8; 4] = Pixel::Rgba8.encode(Colour::Text)[..4].try_into().unwrap();
        // Scale 1 at this height: the panel from (6, 6), 304 wide.
        assert_eq!(at(MARGIN, MARGIN), back_colour, "the panel's corner");
        assert_eq!(at(0, 0), [0; 4], "outside the panel");
        let panel =
            (MARGIN..MARGIN + 100).flat_map(|y| (MARGIN..MARGIN + 300).map(move |x| (x, y)));
        assert!(
            panel.into_iter().any(|(x, y)| at(x, y) == text),
            "text in the panel"
        );
    }

    /// Writes a sample panel to the BMP file `VICTORY_OVERLAY_PREVIEW` names
    /// (`cargo test -p victory-gpu overlay -- --ignored`), to look at it.
    #[test]
    #[ignore]
    fn preview() {
        let Some(path) = std::env::var_os("VICTORY_OVERLAY_PREVIEW") else {
            return;
        };
        let times: VecDeque<f32> = (0..HISTORY)
            .map(|k| 33.3 + if k % 37 == 0 { 30.0 } else { (k % 7) as f32 })
            .collect();
        let second = Second {
            draws: 2513.0,
            prepare_ms: 12.3,
            gpu_wait_ms: 18.2,
        };
        let system = System {
            cpu: Some(37.5),
            private: 14 << 30,
            working: 11 << 30,
            ram: (20 << 30, 64 << 30),
            vram: Some((8 << 30, 15 << 30, 1 << 29)),
        };
        let pad = Pad {
            buttons: 0x4 | 0x800,
            triggers: [0.0, 1.0],
            left: [0.0, 0.98],
            right: [-0.25, 0.0],
        };
        SKIPPED[0].store(true, Relaxed);
        let lines = lines(&times, &second, &system, Some(&pad));
        SKIPPED[0].store(false, Relaxed);
        let image = Image::panel(2, &lines, &times);
        let (w, h) = (image.width, image.height);
        let mut rows = vec![0u8; (w * h * 4) as usize];
        image.write(Pixel::Bgra8, (w * 4) as usize, &mut rows);
        // A 32-bit BMP, rows bottom up.
        let mut bmp = Vec::new();
        let size = 54 + rows.len() as u32;
        bmp.extend(b"BM");
        for v in [size, 0, 54, 40] {
            bmp.extend(v.to_le_bytes());
        }
        bmp.extend((w as i32).to_le_bytes());
        bmp.extend((h as i32).to_le_bytes());
        bmp.extend(1u16.to_le_bytes());
        bmp.extend(32u16.to_le_bytes());
        for v in [0u32, rows.len() as u32, 2835, 2835, 0, 0] {
            bmp.extend(v.to_le_bytes());
        }
        for row in rows.chunks((w * 4) as usize).rev() {
            bmp.extend(row);
        }
        std::fs::write(path, bmp).unwrap();
    }

    /// Each back buffer format gets the colours in its own encoding.
    #[test]
    fn colours_are_encoded_per_format() {
        assert_eq!(Pixel::Rgba8.encode(Colour::Text)[..4], [235, 235, 235, 255]);
        let bad = Pixel::Bgra8.encode(Colour::Bad);
        assert_eq!(bad[..4], [77, 89, 242, 255], "blue, green, red");
        let packed =
            u32::from_le_bytes(Pixel::Rgb10a2.encode(Colour::Back)[..4].try_into().unwrap());
        assert_eq!(packed >> 30, 3, "opaque");
        assert_eq!(half(1.0), 0x3C00);
        assert_eq!(half(0.5), 0x3800);
        let image = Image {
            width: 2,
            height: 1,
            scale: 1,
            pixels: vec![Colour::Back, Colour::Text],
        };
        let mut out = vec![0u8; 256];
        image.write(Pixel::Rgba16f, 256, &mut out);
        assert_eq!(
            out[8..16],
            Pixel::Rgba16f.encode(Colour::Text),
            "the second pixel"
        );
    }
}
