//! Where the title's HTTP requests and WebSockets go instead of the network:
//! a server on loopback, in this process. `net` (WinHTTP) and `xcurl`
//! (curl) send each request for a host that is not loopback here, its
//! origin (scheme, host and port) carried in the path ([`redirected`]).
//!
//! Every request is logged with its full URL, headers and body, every
//! WebSocket message too (`victory_hle::net` target). The server then
//! answers as a console without a network would see a service that is down
//! (HTTP 503; WebSockets are accepted and get nothing), or, where
//! `VICTORY_NET_FORWARD` says so, relays to a real server through the
//! host's WinHTTP:
//!
//! * `origin`: each request to the host the title named;
//! * `http://host:port` or `https://host:port`: every request to that one
//!   server (the title's path kept, its host in `X-Victory-Origin`), e.g. a
//!   server of your own or a recording proxy.
//!
//! Forwarding sends what the title sends, from this machine, to whichever
//! server is named: off unless set.
#![allow(non_snake_case)]

use std::ffi::c_void;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::OnceLock;

use crate::modules::files::host;

/// The path prefix that carries a request's origin.
const PREFIX: &str = "/__victory/";

/// The server's port, the server started on first use (None if loopback
/// cannot be bound).
pub(crate) fn port() -> Option<u16> {
    static PORT: OnceLock<Option<u16>> = OnceLock::new();
    *PORT.get_or_init(|| {
        let listener = match TcpListener::bind(("127.0.0.1", 0)) {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(error = %e, "no loopback server for the title's requests: they fail");
                return None;
            }
        };
        let port = listener.local_addr().ok()?.port();
        tracing::info!(port, forward = ?forward(), "the title's HTTP requests and WebSockets go to a loopback server");
        std::thread::Builder::new()
            .name("victory net server".into())
            .spawn(move || {
                for stream in listener.incoming().flatten() {
                    let _ = std::thread::Builder::new()
                        .name("victory net connection".into())
                        .spawn(move || serve(stream));
                }
            })
            .ok()?;
        Some(port)
    })
}

/// The path that carries `scheme://host:port` + `path` to the server.
pub(crate) fn redirected(scheme: &str, host: &str, port: u16, path: &str) -> String {
    let path = if path.starts_with('/') {
        path.to_owned()
    } else {
        format!("/{path}")
    };
    format!("{PREFIX}{scheme}/{host}:{port}{path}")
}

/// Where requests go on from the server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Forward {
    /// Nowhere: the server answers (HTTP 503; WebSockets get nothing).
    None,
    /// The host the title named.
    Origin,
    /// One server: scheme, host, port.
    To(String, String, u16),
}

/// Where `request` goes on: [`forward`] (tests name it per request).
fn forward_for(request: &Request) -> Forward {
    #[cfg(test)]
    if let Some(v) = request.header(TEST_FORWARD) {
        return parse_forward(v);
    }
    let _ = request;
    forward()
}

/// Tests: a request's own forward target.
#[cfg(test)]
const TEST_FORWARD: &str = "X-Victory-Test-Forward";

/// `VICTORY_NET_FORWARD`, read once.
pub(crate) fn forward() -> Forward {
    static FORWARD: OnceLock<Forward> = OnceLock::new();
    FORWARD
        .get_or_init(|| parse_forward(&std::env::var("VICTORY_NET_FORWARD").unwrap_or_default()))
        .clone()
}

fn parse_forward(value: &str) -> Forward {
    let value = value.trim();
    match value {
        "" | "off" | "0" => Forward::None,
        "origin" => Forward::Origin,
        url => match split_url(url) {
            Some((scheme, host, port, _)) => Forward::To(scheme, host, port),
            None => {
                tracing::warn!(
                    value = url,
                    "VICTORY_NET_FORWARD: not origin or a URL; not forwarding"
                );
                Forward::None
            }
        },
    }
}

/// `scheme://host[:port][path]`: its parts, the port defaulted by scheme.
pub(crate) fn split_url(url: &str) -> Option<(String, String, u16, String)> {
    let (scheme, rest) = url.split_once("://")?;
    let scheme = scheme.to_ascii_lowercase();
    let default = match scheme.as_str() {
        "http" | "ws" => 80,
        "https" | "wss" => 443,
        _ => return None,
    };
    let (authority, path) = match rest.find(['/', '?']) {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let authority = authority.rsplit_once('@').map_or(authority, |(_, a)| a);
    let (host, port) = match authority.rsplit_once(':') {
        Some((h, p)) if !h.ends_with(']') || authority.starts_with('[') => match p.parse() {
            Ok(p) => (h, p),
            Err(_) => (authority, default),
        },
        _ => (authority, default),
    };
    (!host.is_empty()).then(|| (scheme, host.to_owned(), port, path.to_owned()))
}

/// A request as it came in.
#[derive(Debug, Clone)]
pub(crate) struct Request {
    pub method: String,
    /// The origin's URL: scheme, host, port, path and query.
    pub url: String,
    pub scheme: String,
    pub host: String,
    pub port: u16,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Requests and WebSocket messages seen so far (most recent last, the last
/// 256 kept), for tests.
#[cfg(test)]
pub(crate) fn seen() -> Vec<String> {
    SEEN.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

static SEEN: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

fn note(line: String) {
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    if seen.len() == 256 {
        seen.remove(0);
    }
    seen.push(line);
}

/// A body or message as text where it is text, else as hex.
pub(crate) fn shown(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(text)
            if !text
                .chars()
                .any(|c| c.is_control() && !matches!(c, '\n' | '\r' | '\t')) =>
        {
            text.to_owned()
        }
        _ => bytes
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn log_request(r: &Request) {
    let headers: Vec<String> = r.headers.iter().map(|(n, v)| format!("{n}: {v}")).collect();
    tracing::info!(
        target: "victory_hle::net",
        method = %r.method,
        url = %r.url,
        headers = %headers.join("\n"),
        bytes = r.body.len(),
        body = %shown(&r.body),
        "the title's HTTP request"
    );
    note(format!("{} {} {}", r.method, r.url, shown(&r.body)));
}

/// One connection: requests until it closes, or a WebSocket.
fn serve(stream: TcpStream) {
    let Ok(mut out) = stream.try_clone() else {
        return;
    };
    let mut reader = BufReader::new(stream);
    while let Some(request) = read_request(&mut reader) {
        log_request(&request);
        let websocket = request
            .header("Upgrade")
            .is_some_and(|u| u.eq_ignore_ascii_case("websocket"));
        if websocket {
            websocket_session(&request, reader, out);
            return;
        }
        let target = forward_for(&request);
        let (status, reason, headers, body) = match target.clone() {
            Forward::None => (
                503,
                "Service Unavailable".to_owned(),
                vec![("X-Victory".to_owned(), "offline".to_owned())],
                Vec::new(),
            ),
            target => match forward_http(&request, &target) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!(target: "victory_hle::net", url = %request.url, error = %e, "forwarding failed");
                    (
                        502,
                        "Bad Gateway".to_owned(),
                        vec![("X-Victory".to_owned(), "forward failed".to_owned())],
                        Vec::new(),
                    )
                }
            },
        };
        if target != Forward::None {
            tracing::info!(target: "victory_hle::net", url = %request.url, status, bytes = body.len(), body = %shown(&body), "the server's response");
        }
        let mut head = format!("HTTP/1.1 {status} {reason}\r\n");
        for (n, v) in &headers {
            if !["content-length", "transfer-encoding", "connection"]
                .contains(&n.to_ascii_lowercase().as_str())
            {
                head.push_str(&format!("{n}: {v}\r\n"));
            }
        }
        head.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        if out
            .write_all(head.as_bytes())
            .and_then(|()| out.write_all(&body))
            .is_err()
        {
            return;
        }
        if request
            .header("Connection")
            .is_some_and(|c| c.eq_ignore_ascii_case("close"))
        {
            return;
        }
    }
}

/// The next request on a connection; None at its end or on a malformed one.
fn read_request(reader: &mut impl BufRead) -> Option<Request> {
    let mut line = String::new();
    if reader.read_line(&mut line).ok()? == 0 {
        return None;
    }
    let mut parts = line.trim_end().splitn(3, ' ');
    let (method, target) = (parts.next()?.to_owned(), parts.next()?.to_owned());
    let mut headers = Vec::new();
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).ok()? == 0 {
            return None;
        }
        let h = h.trim_end();
        if h.is_empty() {
            break;
        }
        if let Some((n, v)) = h.split_once(':') {
            headers.push((n.trim().to_owned(), v.trim().to_owned()));
        }
    }
    let get = |name: &str| {
        headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
    };
    let body =
        if get("Transfer-Encoding").is_some_and(|t| t.to_ascii_lowercase().contains("chunked")) {
            read_chunked(reader)?
        } else {
            let n: usize = get("Content-Length")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            let mut body = vec![0; n];
            reader.read_exact(&mut body).ok()?;
            body
        };
    // The origin from the path; a request without one names this server.
    let (scheme, host, port, path) = match target.strip_prefix(PREFIX).and_then(|rest| {
        let (scheme, rest) = rest.split_once('/')?;
        let i = rest.find(['/', '?']).unwrap_or(rest.len());
        let (authority, path) = rest.split_at(i);
        let (host, port) = authority.rsplit_once(':')?;
        Some((
            scheme.to_owned(),
            host.to_owned(),
            port.parse().ok()?,
            if path.is_empty() {
                "/".to_owned()
            } else {
                path.to_owned()
            },
        ))
    }) {
        Some(o) => o,
        None => (
            "http".to_owned(),
            get("Host").unwrap_or_default(),
            80,
            target.clone(),
        ),
    };
    let default = if matches!(scheme.as_str(), "https" | "wss") {
        443
    } else {
        80
    };
    let shown_port = if port == default {
        String::new()
    } else {
        format!(":{port}")
    };
    let shown_scheme = if get("Upgrade").is_some_and(|u| u.eq_ignore_ascii_case("websocket")) {
        if scheme == "https" {
            "wss"
        } else if scheme == "http" {
            "ws"
        } else {
            scheme.as_str()
        }
    } else {
        scheme.as_str()
    };
    Some(Request {
        method,
        url: format!("{shown_scheme}://{host}{shown_port}{path}"),
        scheme,
        host,
        port,
        path,
        headers,
        body,
    })
}

fn read_chunked(reader: &mut impl BufRead) -> Option<Vec<u8>> {
    let mut body = Vec::new();
    loop {
        let mut size = String::new();
        reader.read_line(&mut size).ok()?;
        let n = usize::from_str_radix(size.trim().split(';').next()?.trim(), 16).ok()?;
        if n == 0 {
            // Trailers, then the blank line.
            loop {
                let mut t = String::new();
                if reader.read_line(&mut t).ok()? == 0 || t.trim().is_empty() {
                    return Some(body);
                }
            }
        }
        let at = body.len();
        body.resize(at + n, 0);
        reader.read_exact(&mut body[at..]).ok()?;
        let mut crlf = [0; 2];
        reader.read_exact(&mut crlf).ok()?;
    }
}

// WebSockets (RFC 6455).

/// The handshake's accept key for the client's `key`.
pub(crate) fn accept_key(key: &str) -> String {
    let digest = sha1(format!("{key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11").as_bytes());
    base64(&digest)
}

fn websocket_session(request: &Request, mut reader: BufReader<TcpStream>, mut out: TcpStream) {
    let Some(key) = request.header("Sec-WebSocket-Key") else {
        return;
    };
    let upstream = match forward_for(request) {
        Forward::None => None,
        target => match Upstream::open(request, &target) {
            Ok(u) => Some(u),
            Err(e) => {
                tracing::warn!(target: "victory_hle::net", url = %request.url, error = %e, "WebSocket forwarding failed");
                let _ = out.write_all(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\n\r\n");
                return;
            }
        },
    };
    let mut head = format!(
        "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n",
        accept_key(key)
    );
    if let Some(p) = request.header("Sec-WebSocket-Protocol") {
        head.push_str(&format!(
            "Sec-WebSocket-Protocol: {}\r\n",
            p.split(',').next().unwrap_or(p).trim()
        ));
    }
    head.push_str("\r\n");
    if out.write_all(head.as_bytes()).is_err() {
        return;
    }
    let url = request.url.clone();
    // The server's messages to the title.
    if let Some(up) = &upstream {
        let (up, mut to_title, url) = (up.clone(), out.try_clone().ok(), url.clone());
        std::thread::spawn(move || {
            while let Some((text, data)) = up.receive() {
                log_message(&url, "from the server", text, &data);
                let Some(t) = to_title.as_mut() else { return };
                if t.write_all(&frame(if text { 1 } else { 2 }, &data))
                    .is_err()
                {
                    return;
                }
            }
            if let Some(t) = to_title.as_mut() {
                let _ = t.write_all(&frame(8, &[]));
            }
        });
    }
    // The title's messages.
    let mut message: Option<(bool, Vec<u8>)> = None;
    while let Some((fin, opcode, data)) = read_frame(&mut reader) {
        match opcode {
            0x8 => {
                let _ = out.write_all(&frame(8, &data[..data.len().min(2)]));
                if let Some(up) = &upstream {
                    up.close();
                }
                tracing::info!(target: "victory_hle::net", %url, "the title closed a WebSocket");
                return;
            }
            0x9 => {
                let _ = out.write_all(&frame(0xA, &data));
            }
            0xA => {}
            _ => {
                let (text, mut bytes) = match (opcode, message.take()) {
                    (0, Some(m)) => m,
                    (op, _) => (op == 1, Vec::new()),
                };
                bytes.extend_from_slice(&data);
                if fin {
                    log_message(&url, "from the title", text, &bytes);
                    if let Some(up) = &upstream {
                        up.send(text, &bytes);
                    }
                } else {
                    message = Some((text, bytes));
                }
            }
        }
    }
}

fn log_message(url: &str, from: &str, text: bool, data: &[u8]) {
    tracing::info!(
        target: "victory_hle::net",
        %url,
        kind = if text { "text" } else { "binary" },
        bytes = data.len(),
        message = %shown(data),
        "WebSocket message {from}"
    );
    note(format!("ws {from} {url} {}", shown(data)));
}

/// A frame from the title (masked, as clients send): FIN, opcode, payload.
fn read_frame(r: &mut impl Read) -> Option<(bool, u8, Vec<u8>)> {
    let mut h = [0; 2];
    r.read_exact(&mut h).ok()?;
    let (fin, opcode, masked) = (h[0] & 0x80 != 0, h[0] & 0xF, h[1] & 0x80 != 0);
    let len = match h[1] & 0x7F {
        126 => {
            let mut b = [0; 2];
            r.read_exact(&mut b).ok()?;
            u64::from(u16::from_be_bytes(b))
        }
        127 => {
            let mut b = [0; 8];
            r.read_exact(&mut b).ok()?;
            u64::from_be_bytes(b)
        }
        n => u64::from(n),
    };
    let mut mask = [0; 4];
    if masked {
        r.read_exact(&mut mask).ok()?;
    }
    let mut data = vec![0; usize::try_from(len).ok().filter(|&n| n <= 256 << 20)?];
    r.read_exact(&mut data).ok()?;
    if masked {
        for (i, b) in data.iter_mut().enumerate() {
            *b ^= mask[i % 4];
        }
    }
    Some((fin, opcode, data))
}

/// A server's frame (unmasked).
fn frame(opcode: u8, data: &[u8]) -> Vec<u8> {
    let mut f = vec![0x80 | opcode];
    match data.len() {
        n if n < 126 => f.push(n as u8),
        n if n <= 0xFFFF => {
            f.push(126);
            f.extend_from_slice(&(n as u16).to_be_bytes());
        }
        n => {
            f.push(127);
            f.extend_from_slice(&(n as u64).to_be_bytes());
        }
    }
    f.extend_from_slice(data);
    f
}

fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [
        0x6745_2301,
        0xEFCD_AB89,
        0x98BA_DCFE,
        0x1032_5476,
        0xC3D2_E1F0,
    ];
    let mut m = data.to_vec();
    let bits = (data.len() as u64) * 8;
    m.push(0x80);
    while m.len() % 64 != 56 {
        m.push(0);
    }
    m.extend_from_slice(&bits.to_be_bytes());
    for block in m.chunks(64) {
        let mut w = [0u32; 80];
        for (i, word) in block.chunks(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, &wi) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A82_7999),
                20..=39 => (b ^ c ^ d, 0x6ED9_EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1B_BCDC),
                _ => (b ^ c ^ d, 0xCA62_C1D6),
            };
            let t = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(wi);
            (e, d, c, b, a) = (d, c, b.rotate_left(30), a, t);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0; 20];
    for (i, x) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&x.to_be_bytes());
    }
    out
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in data.chunks(3) {
        let n = u32::from(c[0]) << 16
            | u32::from(*c.get(1).unwrap_or(&0)) << 8
            | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            s.push(if i <= c.len() {
                T[(n >> (18 - 6 * i) & 63) as usize] as char
            } else {
                '='
            });
        }
    }
    s
}

// Forwarding, through the host's WinHTTP (the title's own WinHTTP calls are
// vetted by `net`; these are ours).

type Handle = *mut c_void;

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(Some(0)).collect()
}

fn last_error() -> String {
    std::io::Error::last_os_error().to_string()
}

/// Where `request` goes for `target`: scheme, host, port.
fn destination(request: &Request, target: &Forward) -> (String, String, u16) {
    match target {
        Forward::To(s, h, p) => (s.clone(), h.clone(), *p),
        _ => (request.scheme.clone(), request.host.clone(), request.port),
    }
}

/// A host WinHTTP session, connection and request; closed on drop.
struct Winhttp(Vec<Handle>);

impl Drop for Winhttp {
    fn drop(&mut self) {
        for &h in self.0.iter().rev() {
            if !h.is_null() {
                host!("winhttp.dll", "WinHttpCloseHandle": extern "system" fn(Handle) -> i32)(h);
            }
        }
    }
}

// SAFETY: WinHTTP handles are usable from any thread.
unsafe impl Send for Winhttp {}
// SAFETY: as above; calls on one handle are not made at once.
unsafe impl Sync for Winhttp {}

/// Opens a request to where `request` goes, sends it (with its body, or
/// set to upgrade to a WebSocket) and receives the response head.
fn send(request: &Request, target: &Forward, websocket: bool) -> Result<Winhttp, String> {
    let (scheme, host, port) = destination(request, target);
    let secure = matches!(scheme.as_str(), "https" | "wss");
    let open = host!("winhttp.dll", "WinHttpOpen": extern "system" fn(*const u16, u32, *const u16, *const u16, u32) -> Handle);
    let connect = host!("winhttp.dll", "WinHttpConnect": extern "system" fn(Handle, *const u16, u16, u32) -> Handle);
    let open_request = host!("winhttp.dll", "WinHttpOpenRequest": extern "system" fn(Handle, *const u16, *const u16, *const u16, *const u16, *const *const u16, u32) -> Handle);
    let mut h = Winhttp(Vec::new());
    // Default proxy settings: the machine's own.
    let session = open(
        wide("VIctory").as_ptr(),
        0,
        std::ptr::null(),
        std::ptr::null(),
        0,
    );
    h.0.push(session);
    if session.is_null() {
        return Err(format!("WinHttpOpen: {}", last_error()));
    }
    let c = connect(session, wide(&host).as_ptr(), port, 0);
    h.0.push(c);
    if c.is_null() {
        return Err(format!("WinHttpConnect {host}:{port}: {}", last_error()));
    }
    const SECURE: u32 = 0x0080_0000;
    let r = open_request(
        c,
        wide(&request.method).as_ptr(),
        wide(&request.path).as_ptr(),
        std::ptr::null(),
        std::ptr::null(),
        std::ptr::null(),
        if secure { SECURE } else { 0 },
    );
    h.0.push(r);
    if r.is_null() {
        return Err(format!("WinHttpOpenRequest: {}", last_error()));
    }
    if websocket {
        const UPGRADE: u32 = 114;
        let set = host!("winhttp.dll", "WinHttpSetOption": extern "system" fn(Handle, u32, *const c_void, u32) -> i32);
        if set(r, UPGRADE, std::ptr::null(), 0) == 0 {
            return Err(format!("WebSocket upgrade option: {}", last_error()));
        }
    }
    let skip = |n: &str| {
        let n = n.to_ascii_lowercase();
        [
            "host",
            "connection",
            "content-length",
            "transfer-encoding",
            "upgrade",
            "x-victory-test-forward",
        ]
        .contains(&n.as_str())
            || n.starts_with("sec-websocket-")
    };
    let mut headers: String = request
        .headers
        .iter()
        .filter(|(n, _)| !skip(n))
        .map(|(n, v)| format!("{n}: {v}\r\n"))
        .collect();
    if matches!(target, Forward::To(..)) {
        headers.push_str(&format!(
            "X-Victory-Origin: {}://{}:{}\r\n",
            request.scheme, request.host, request.port
        ));
    }
    let headers = wide(&headers);
    let send_request = host!("winhttp.dll", "WinHttpSendRequest": extern "system" fn(Handle, *const u16, u32, *const c_void, u32, u32, usize) -> i32);
    let body = &request.body;
    let len = u32::try_from(body.len()).map_err(|_| "a body over 4 GiB")?;
    if send_request(
        r,
        headers.as_ptr(),
        u32::MAX,
        body.as_ptr().cast(),
        len,
        len,
        0,
    ) == 0
    {
        return Err(format!(
            "WinHttpSendRequest to {scheme}://{host}:{port}: {}",
            last_error()
        ));
    }
    let receive = host!("winhttp.dll", "WinHttpReceiveResponse": extern "system" fn(Handle, *mut c_void) -> i32);
    if receive(r, std::ptr::null_mut()) == 0 {
        return Err(format!("WinHttpReceiveResponse: {}", last_error()));
    }
    Ok(h)
}

/// The status, reason and headers of a received response.
fn response_head(r: Handle) -> (u16, String, Vec<(String, String)>) {
    let query = host!("winhttp.dll", "WinHttpQueryHeaders": extern "system" fn(Handle, u32, *const u16, *mut c_void, *mut u32, *mut u32) -> i32);
    const RAW_HEADERS_CRLF: u32 = 22;
    let mut len = 0u32;
    query(
        r,
        RAW_HEADERS_CRLF,
        std::ptr::null(),
        std::ptr::null_mut(),
        &mut len,
        std::ptr::null_mut(),
    );
    let mut buffer = vec![0u16; len as usize / 2 + 1];
    if query(
        r,
        RAW_HEADERS_CRLF,
        std::ptr::null(),
        buffer.as_mut_ptr().cast(),
        &mut len,
        std::ptr::null_mut(),
    ) == 0
    {
        return (502, "Bad Gateway".into(), Vec::new());
    }
    let text = String::from_utf16_lossy(&buffer[..len as usize / 2]);
    let mut lines = text.split("\r\n");
    let status_line = lines.next().unwrap_or_default();
    let mut parts = status_line.splitn(3, ' ');
    let _version = parts.next();
    let status = parts.next().and_then(|s| s.parse().ok()).unwrap_or(502);
    let reason = parts.next().unwrap_or("").to_owned();
    let headers = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(n, v)| (n.trim().to_owned(), v.trim().to_owned()))
        .collect();
    (status, reason, headers)
}

/// A response: status, reason, headers, body.
type Response = (u16, String, Vec<(String, String)>, Vec<u8>);

fn forward_http(request: &Request, target: &Forward) -> Result<Response, String> {
    let h = send(request, target, false)?;
    let r = h.0[2];
    let (status, reason, headers) = response_head(r);
    let read = host!("winhttp.dll", "WinHttpReadData": extern "system" fn(Handle, *mut c_void, u32, *mut u32) -> i32);
    let mut body = Vec::new();
    let mut chunk = vec![0u8; 64 << 10];
    loop {
        let mut n = 0u32;
        if read(r, chunk.as_mut_ptr().cast(), chunk.len() as u32, &mut n) == 0 {
            return Err(format!("WinHttpReadData: {}", last_error()));
        }
        if n == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..n as usize]);
    }
    Ok((status, reason, headers, body))
}

/// A forwarded WebSocket: the host's WebSocket handle, and its request's
/// handles (kept until it closes).
#[derive(Clone)]
struct Upstream(std::sync::Arc<(Winhttp, Winhttp)>);

impl Upstream {
    fn open(request: &Request, target: &Forward) -> Result<Upstream, String> {
        let h = send(request, target, true)?;
        let (status, _, _) = response_head(h.0[2]);
        if status != 101 {
            return Err(format!("the server answered {status}, not 101"));
        }
        let complete = host!("winhttp.dll", "WinHttpWebSocketCompleteUpgrade": extern "system" fn(Handle, usize) -> Handle);
        let ws = complete(h.0[2], 0);
        if ws.is_null() {
            return Err(format!("WinHttpWebSocketCompleteUpgrade: {}", last_error()));
        }
        Ok(Upstream(std::sync::Arc::new((Winhttp(vec![ws]), h))))
    }

    fn ws(&self) -> Handle {
        self.0.0.0[0]
    }

    fn send(&self, text: bool, data: &[u8]) {
        let send = host!("winhttp.dll", "WinHttpWebSocketSend": extern "system" fn(Handle, u32, *const c_void, u32) -> u32);
        const BINARY: u32 = 0;
        const UTF8: u32 = 2;
        send(
            self.ws(),
            if text { UTF8 } else { BINARY },
            data.as_ptr().cast(),
            data.len() as u32,
        );
    }

    /// The next whole message: whether it is text, and its bytes; None once
    /// closed.
    fn receive(&self) -> Option<(bool, Vec<u8>)> {
        let receive = host!("winhttp.dll", "WinHttpWebSocketReceive": extern "system" fn(Handle, *mut c_void, u32, *mut u32, *mut u32) -> u32);
        let mut message = Vec::new();
        let mut chunk = vec![0u8; 64 << 10];
        loop {
            let (mut n, mut kind) = (0u32, 0u32);
            if receive(
                self.ws(),
                chunk.as_mut_ptr().cast(),
                chunk.len() as u32,
                &mut n,
                &mut kind,
            ) != 0
            {
                return None;
            }
            message.extend_from_slice(&chunk[..n as usize]);
            match kind {
                0 => return Some((false, message)),
                2 => return Some((true, message)),
                1 | 3 => {}
                _ => return None,
            }
        }
    }

    fn close(&self) {
        let close = host!("winhttp.dll", "WinHttpWebSocketClose": extern "system" fn(Handle, u16, *const c_void, u32) -> u32);
        close(self.ws(), 1000, std::ptr::null(), 0);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Whether `line` was logged.
    pub(crate) fn seen_has(line: &str) -> bool {
        seen().iter().any(|l| l == line)
    }

    #[test]
    fn urls_split_into_their_parts() {
        let s = |u: &str| split_url(u).map(|(s, h, p, path)| format!("{s} {h} {p} {path}"));
        assert_eq!(
            s("https://api.example.com/v1/x?y=1").as_deref(),
            Some("https api.example.com 443 /v1/x?y=1")
        );
        assert_eq!(
            s("http://example.com:8080").as_deref(),
            Some("http example.com 8080 /")
        );
        assert_eq!(
            s("wss://user@chat.example.com:9000/s").as_deref(),
            Some("wss chat.example.com 9000 /s")
        );
        assert_eq!(s("ftp://example.com"), None);
        assert_eq!(parse_forward(""), Forward::None);
        assert_eq!(parse_forward("origin"), Forward::Origin);
        assert_eq!(
            parse_forward("http://127.0.0.1:9"),
            Forward::To("http".into(), "127.0.0.1".into(), 9)
        );
    }

    /// RFC 6455's example handshake.
    #[test]
    fn the_accept_key_is_the_rfcs() {
        assert_eq!(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
        assert_eq!(base64(b"ab"), "YWI=");
    }

    fn connect() -> TcpStream {
        TcpStream::connect(("127.0.0.1", port().expect("a loopback server"))).unwrap()
    }

    /// A request for a host on the network, as `net` and `xcurl` send it:
    /// logged with its full URL and body, answered 503.
    #[test]
    fn requests_are_logged_with_their_url_and_body_and_refused() {
        let mut s = connect();
        let path = redirected("https", "api.example.com", 443, "/v1/session?x=1");
        let body = b"{\"user\":\"offline\"}";
        write!(s, "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len()).unwrap();
        s.write_all(body).unwrap();
        let mut answer = String::new();
        s.read_to_string(&mut answer).unwrap();
        assert!(answer.starts_with("HTTP/1.1 503"), "{answer}");
        assert!(
            seen()
                .iter()
                .any(|l| l == "POST https://api.example.com/v1/session?x=1 {\"user\":\"offline\"}"),
            "{:?}",
            seen()
        );
    }

    /// Forwarded to a server of one's own: the title's path and body reach
    /// it, with the origin in `X-Victory-Origin`, and its answer comes back.
    #[test]
    fn requests_forward_to_a_server_and_its_answer_comes_back() {
        let upstream = TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let up = upstream.local_addr().unwrap().port();
        let got = std::thread::spawn(move || {
            let (s, _) = upstream.accept().unwrap();
            let mut out = s.try_clone().unwrap();
            let request = read_request(&mut BufReader::new(s)).unwrap();
            out.write_all(
                b"HTTP/1.1 200 OK
Content-Type: text/plain
Content-Length: 5

hello",
            )
            .unwrap();
            request
        });
        let mut s = connect();
        let path = redirected("https", "api.example.com", 443, "/v1/thing");
        write!(
            s,
            "POST {path} HTTP/1.1
{TEST_FORWARD}: http://127.0.0.1:{up}
Content-Length: 3
Connection: close

abc"
        )
        .unwrap();
        let mut answer = String::new();
        s.read_to_string(&mut answer).unwrap();
        assert!(answer.starts_with("HTTP/1.1 200"), "{answer}");
        assert!(answer.ends_with("hello"), "{answer}");
        let request = got.join().unwrap();
        assert_eq!(
            (
                request.method.as_str(),
                request.path.as_str(),
                request.body.as_slice()
            ),
            ("POST", "/v1/thing", b"abc".as_slice())
        );
        assert_eq!(
            request.header("X-Victory-Origin"),
            Some("https://api.example.com:443")
        );
        assert_eq!(request.header(TEST_FORWARD), None, "not passed on");
    }

    /// A WebSocket forwarded: the title's messages go on to the server (here
    /// our own, unforwarded, which logs them as it would the title's).
    #[test]
    fn websocket_messages_forward_to_the_server() {
        let ours = port().unwrap();
        let mut s = connect();
        let path = redirected("https", "chat.example.com", 443, "/relay");
        write!(
            s,
            "GET {path} HTTP/1.1
{TEST_FORWARD}: http://127.0.0.1:{ours}
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13

"
        )
        .unwrap();
        let mut reader = BufReader::new(s.try_clone().unwrap());
        let mut status = String::new();
        reader.read_line(&mut status).unwrap();
        assert!(status.starts_with("HTTP/1.1 101"), "{status}");
        let mask = [9u8, 8, 7, 6];
        let mut f = vec![0x81, 0x80 | 5];
        f.extend_from_slice(&mask);
        f.extend(b"relay".iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
        s.write_all(&f).unwrap();
        let wanted = format!("ws from the title ws://127.0.0.1:{ours}/relay relay");
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !seen_has(&wanted) && std::time::Instant::now() < deadline {
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        assert!(seen_has(&wanted), "{:?}", seen());
        assert!(
            seen_has("ws from the title wss://chat.example.com/relay relay"),
            "logged as the title's too"
        );
    }

    /// A chunked body arrives whole.
    #[test]
    fn chunked_bodies_are_logged_whole() {
        let mut s = connect();
        let path = redirected("http", "example.com", 8080, "/upload");
        write!(s, "PUT {path} HTTP/1.1\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n4\r\nabcd\r\n3\r\nefg\r\n0\r\n\r\n").unwrap();
        let mut answer = String::new();
        s.read_to_string(&mut answer).unwrap();
        assert!(
            seen()
                .iter()
                .any(|l| l == "PUT http://example.com:8080/upload abcdefg"),
            "{:?}",
            seen()
        );
    }

    /// A WebSocket is accepted; the title's messages, fragmented and masked,
    /// are logged whole; a ping gets its pong; a close is answered.
    #[test]
    fn websocket_messages_are_logged() {
        let mut s = connect();
        let path = redirected("https", "chat.example.com", 443, "/socket");
        write!(s, "GET {path} HTTP/1.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n").unwrap();
        let mut reader = BufReader::new(s.try_clone().unwrap());
        let mut head = String::new();
        loop {
            let mut l = String::new();
            reader.read_line(&mut l).unwrap();
            head.push_str(&l);
            if l == "\r\n" {
                break;
            }
        }
        assert!(head.starts_with("HTTP/1.1 101"), "{head}");
        assert!(head.contains("s3pPLMBiTxaQ9kYGzzhZRbK+xOo="));
        let masked = |fin: bool, opcode: u8, data: &[u8]| {
            let mask = [1u8, 2, 3, 4];
            let mut f = vec![u8::from(fin) << 7 | opcode, 0x80 | data.len() as u8];
            f.extend_from_slice(&mask);
            f.extend(data.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
            f
        };
        s.write_all(&masked(false, 1, b"hello ")).unwrap();
        s.write_all(&masked(true, 0, b"world")).unwrap();
        s.write_all(&masked(true, 9, b"p")).unwrap();
        let (_, opcode, data) = read_frame(&mut reader).unwrap();
        assert_eq!(
            (opcode, data.as_slice()),
            (0xA, b"p".as_slice()),
            "the pong"
        );
        s.write_all(&masked(true, 8, &1000u16.to_be_bytes()))
            .unwrap();
        let (_, opcode, _) = read_frame(&mut reader).unwrap();
        assert_eq!(opcode, 8, "the close");
        assert!(
            seen()
                .iter()
                .any(|l| l == "ws from the title wss://chat.example.com/socket hello world"),
            "{:?}",
            seen()
        );
    }
}
