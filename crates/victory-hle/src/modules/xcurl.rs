//! `XCurl.dll`, the curl titles ship (libHttpClient's HTTP on consoles
//! until 2026-08; its WebSockets go through WinHTTP, `net`). Its own DLL
//! still loads and does the work (the module wraps it, [`crate::wraps`]):
//! only the URL a request is given is ours to see. One for a host that is
//! not loopback becomes a plain HTTP URL on our server (`net_server`), its
//! origin in the path, where it is logged with its body and answered or
//! forwarded; curl's sockets then reach loopback only, as `net` allows.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::OnceLock;

use crate::{HleModule, hle_export};

/// `CURLOPT_URL`: `CURLOPTTYPE_STRINGPOINT` + 2.
const CURLOPT_URL: i32 = 10_002;
/// `CURLE_FAILED_INIT`.
const FAILED_INIT: i32 = 2;

type Setopt = extern "C" fn(*mut c_void, i32, usize) -> i32;

/// The title's own `curl_easy_setopt`.
fn real() -> Option<Setopt> {
    static REAL: OnceLock<Option<usize>> = OnceLock::new();
    let f = *REAL.get_or_init(|| {
        let base = crate::runtime::image_named("xcurl.dll")?;
        // SAFETY: a module the loader mapped; a NUL-terminated name.
        let f = unsafe {
            windows_sys::Win32::System::LibraryLoader::GetProcAddress(
                base as _,
                c"curl_easy_setopt".as_ptr().cast(),
            )
        }?;
        Some(f as usize)
    });
    // SAFETY: curl's own export of this signature (a variadic one: its third
    // argument goes in a register as a 64-bit value, as this declares it).
    f.map(|f| unsafe { std::mem::transmute::<usize, Setopt>(f) })
}

/// The URL a request for `url` goes to instead: ours, where its host is not
/// loopback and our server runs.
pub(crate) fn rewritten(url: &str) -> Option<String> {
    let (scheme, host, port, path) = super::net_server::split_url(url)?;
    if super::net::loopback_name(&host) {
        return None;
    }
    let ours = super::net_server::port()?;
    Some(format!(
        "http://127.0.0.1:{ours}{}",
        super::net_server::redirected(&scheme, &host, port, &path)
    ))
}

/// Every option passes to curl; a request's URL first goes through
/// [`rewritten`].
pub extern "C" fn curl_easy_setopt(handle: *mut c_void, option: i32, param: usize) -> i32 {
    let Some(real) = real() else {
        tracing::warn!("curl_easy_setopt: the title's XCurl.dll is not loaded");
        return FAILED_INIT;
    };
    if option == CURLOPT_URL && param != 0 {
        // SAFETY: curl takes a NUL-terminated string for this option.
        let url = unsafe { CStr::from_ptr(param as *const c_char) }
            .to_string_lossy()
            .into_owned();
        if let Some(ours) = rewritten(&url).and_then(|u| CString::new(u).ok()) {
            tracing::debug!(%url, "a curl request goes to our server");
            // curl copies the string.
            return real(handle, option, ours.as_ptr() as usize);
        }
    }
    real(handle, option, param)
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xcurl",
        dlls: &["xcurl.dll"],
        exports: vec![hle_export!(curl_easy_setopt)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_for_the_network_go_to_our_server() {
        let port = super::super::net_server::port().unwrap();
        assert_eq!(
            rewritten("https://api.example.com/v1/x?y=1").as_deref(),
            Some(
                format!("http://127.0.0.1:{port}/__victory/https/api.example.com:443/v1/x?y=1")
                    .as_str()
            )
        );
        assert_eq!(rewritten("http://localhost:8080/x"), None, "loopback stays");
        assert_eq!(rewritten("not a url"), None);
    }
}
