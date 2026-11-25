//! The conformance title (tests/fixtures/conformance, docs/acceptance.md)
//! run through VIctory's loader and runtime, as `victory run` runs a title:
//! its checks are what Microsoft's public GDK reference says each call does.
//! No game data and no SDK.
#![cfg(windows)]

use std::path::{Path, PathBuf};
use std::process::Command;

use victory_hle::{HleResolver, HostPolicy};
use victory_loader::{UnimplementedPolicy, prepare};

/// Checks VIctory is known to answer differently from the public reference,
/// each with why: reported, not failed. A check here that passes fails the
/// test, so the list stays true.
const KNOWN: &[(&str, &str)] = &[
    (
        "xasync: an async call with no queue and no process queue is E_NO_TASK_QUEUE",
        "VIctory runs the call anyway; XTaskQueueSetCurrentProcessTaskQueue's \
         page says APIs needing a queue then fail with E_NO_TASK_QUEUE",
    ),
    (
        "xsystem: a retail console id is the documented zero id",
        "VIctory returns 38 digits ending in 1; the reference gives \
         00000000.00000000.00000000.00000000.00 for retail and PC",
    ),
    (
        "dstorage: a queue filled to half its capacity is not submitted on its own",
        "VIctory submits once half the capacity is enqueued; the overview \
         says an Enqueue that fills more than half submits",
    ),
    (
        "xuser: XUserFindUserByLocalId of no user is HRESULT_FROM_WIN32(ERROR_NOT_FOUND)",
        "VIctory returns E_GAMEUSER_USER_NOT_FOUND (0x89245104); the \
         XUserFindUserByLocalId page gives HRESULT_FROM_WIN32(ERROR_NOT_FOUND)",
    ),
    (
        "xuser: an invalid gamertag component is E_INVALIDARG",
        "VIctory answers any component other than ModernSuffix with the \
         gamertag; the XUserGetGamertag page gives E_INVALIDARG",
    ),
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Builds the fixture into its own target dir (the outer cargo holds the
/// lock on the main one while tests run).
fn build_fixture() -> PathBuf {
    let manifest = workspace_root().join("tests/fixtures/conformance/Cargo.toml");
    let target = Path::new(env!("CARGO_TARGET_TMPDIR")).join("conformance");
    let status = Command::new(env!("CARGO"))
        .args(["build", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .arg("--target-dir")
        .arg(&target)
        .status()
        .expect("running cargo");
    assert!(status.success(), "building the conformance title failed");
    target.join("debug").join("conformance.dll")
}

/// The title loaded and bound once per test process (the runtime's state
/// is global: one title at a time); `call(name)` runs one of its report
/// exports.
fn call(name: &str) -> String {
    static EXPORTS: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();
    static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let &(run, bench) = EXPORTS.get_or_init(|| {
        let dll = build_fixture();
        let original = std::fs::read(&dll).unwrap();
        let prepared = prepare(&original).unwrap();
        let images = tempfile::tempdir().unwrap().keep();
        let image = victory_loader::load(&prepared, &images).unwrap();
        victory_hle::runtime::set_main_image(image.base());
        victory_hle::runtime::set_image_path(image.base(), &dll);
        // Persistent local storage (game saves live beside it), as `victory
        // run` gives a title.
        let data = tempfile::tempdir().unwrap().keep();
        // `victory run` makes the folder before the title starts.
        std::fs::create_dir_all(data.join("storage")).unwrap();
        victory_hle::runtime::set_title(victory_hle::runtime::Title {
            id: 0x1234_5678,
            storage: data.join("storage"),
            storage_bytes: 64 << 20,
            ..Default::default()
        });
        // The console's drives: the package at G: (holding the file the
        // title checks), temporary storage at T:.
        let package = data.join("package");
        std::fs::create_dir_all(&package).unwrap();
        std::fs::write(
            package.join("conformance.txt"),
            b"conformance: public documentation only
",
        )
        .unwrap();
        let temp = data.join("temp");
        std::fs::create_dir_all(&temp).unwrap();
        victory_hle::modules::files::mount_at('G', &package);
        victory_hle::modules::files::mount_at('T', &temp);
        let resolver = HleResolver::new(HostPolicy::default());
        let report = image
            .bind_imports(&resolver, UnimplementedPolicy::ReturnZero)
            .unwrap();
        assert!(
            report.missing.is_empty(),
            "imports VIctory does not serve: {:?}",
            report.missing.iter().map(|m| &m.symbol).collect::<Vec<_>>()
        );
        // SAFETY: its imports are bound; it depends on no title DLL.
        unsafe { image.attach() }.unwrap();
        let exports = (
            image.export("conformance_run").expect("conformance_run"),
            image
                .export("conformance_bench")
                .expect("conformance_bench"),
        );
        // Stays loaded for the rest of the process.
        std::mem::forget(image);
        exports
    });
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let at = if name == "conformance_bench" {
        bench
    } else {
        run
    };
    // SAFETY: the exports' signature in the fixture; the report stays valid
    // until the next call, which the lock holds off.
    unsafe {
        let f: extern "system" fn(*mut usize) -> *const u8 = std::mem::transmute(at);
        let mut len = 0;
        let report = f(&mut len);
        String::from_utf8_lossy(std::slice::from_raw_parts(report, len)).into_owned()
    }
}

/// The streaming workload (docs/acceptance.md): throughput, printed. Run
/// with `cargo test --release -p victory-hle --test conformance -- /// --ignored --nocapture`.
#[test]
#[ignore = "a benchmark: run on demand"]
fn workloads() {
    println!("{}", call("conformance_bench"));
}

#[test]
fn the_conformance_title_runs_through_victory() {
    let tap = call("conformance_run");
    println!("{tap}");

    let failed: Vec<&str> = tap
        .lines()
        .filter_map(|l| l.strip_prefix("not ok "))
        .filter_map(|l| l.split_once(" - ").map(|(_, name)| name))
        .collect();
    let passed: Vec<&str> = tap
        .lines()
        .filter_map(|l| l.strip_prefix("ok "))
        .filter_map(|l| l.split_once(" - ").map(|(_, name)| name))
        .collect();
    let known = |name: &str| KNOWN.iter().find(|(k, _)| *k == name);
    for name in &failed {
        if let Some((_, why)) = known(name) {
            println!("known divergence: {name}: {why}");
        }
    }
    let unexpected: Vec<&&str> = failed.iter().filter(|n| known(n).is_none()).collect();
    let fixed: Vec<&str> = KNOWN
        .iter()
        .map(|(k, _)| *k)
        .filter(|k| passed.contains(k))
        .collect();
    assert!(passed.len() + failed.len() > 0, "no checks ran:\n{tap}");
    assert!(unexpected.is_empty(), "checks that failed: {unexpected:#?}");
    assert!(
        fixed.is_empty(),
        "known divergences that now pass (take them off KNOWN): {fixed:#?}"
    );
}
