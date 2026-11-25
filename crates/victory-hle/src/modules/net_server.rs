//! Where the title's HTTP requests and WebSockets go instead of the network:
//! a server on loopback, in this process. `net` (WinHTTP) and `xcurl`
//! (curl) send each request for a host that is not loopback here, its
//! origin (scheme, host and port) carried in the path ([`redirected`]).
//!
//! Every request is logged with its full URL, headers and body, every
//! WebSocket message too (`victory_hle::net` target). The server then
//! answers as a console without a network would see a service that is down
//! (HTTP 503; WebSockets are accepted and get nothing). Nothing the title
//! sends leaves this machine.
#![allow(non_snake_case)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::OnceLock;

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
        tracing::info!(port, "the title's HTTP requests and WebSockets go to a loopback server");
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
        let (status, reason, headers, body): (u16, String, Vec<(String, String)>, Vec<u8>) = (
            503,
            "Service Unavailable".to_owned(),
            vec![("X-Victory".to_owned(), "offline".to_owned())],
            Vec::new(),
        );
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
    // The title's messages.
    let mut message: Option<(bool, Vec<u8>)> = None;
    while let Some((fin, opcode, data)) = read_frame(&mut reader) {
        match opcode {
            0x8 => {
                let _ = out.write_all(&frame(8, &data[..data.len().min(2)]));
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
