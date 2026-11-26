//! `prebuild` (`cargo prebuild`): the README's warm-up step without its
//! paths. The files the title's shaders are in are found in the game
//! folder (its shader archive, its executable and the DLLs beside it), and
//! the shader cache and pipeline library are made where `run` reads them:
//! both go through the same configuration (`victory.toml` in the folder it
//! is started from, or `VICTORY_CONFIG`), which is shown first.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Makes the shader cache, and with `pipelines` the pipeline library, for
/// the title in `game_dir` (default: the configured game folder).
pub(crate) fn run(game_dir: Option<PathBuf>, jobs: Option<usize>, pipelines: bool) -> Result<()> {
    let cfg = victory_core::Config::load()?;
    match &cfg.source {
        Some(file) => println!("config         {}", file.display()),
        None => println!(
            "config         none in {} (defaults and VICTORY_* variables only): run this \
             where you run the title, or set VICTORY_CONFIG, so that both use one cache",
            std::env::current_dir().unwrap_or_default().display()
        ),
    }
    let game_dir = match game_dir {
        Some(dir) => dir,
        None => cfg.require_game_dir()?.to_owned(),
    };
    let files = title_files(&game_dir)?;
    println!("shader cache   {}", cfg.cache_subdir("shaders")?.display());
    if pipelines {
        println!(
            "pipelines      {}",
            cfg.cache_subdir("pipelines")?.display()
        );
    }
    for file in &files {
        println!("reads          {}", file.display());
    }
    crate::translate::warm(files, jobs, pipelines)
}

/// The files of the title in `game_dir` that hold its shaders: its shader
/// archive (`shaders.rpf`, which holds the shader pack the pipelines come
/// from), its executable (the first its manifest names that is there, else
/// the folder's only `.exe`) and the DLLs beside it.
pub(crate) fn title_files(game_dir: &Path) -> Result<Vec<PathBuf>> {
    anyhow::ensure!(game_dir.is_dir(), "{} is not a folder", game_dir.display());
    let manifest =
        std::fs::read_to_string(game_dir.join("MicrosoftGame.config")).unwrap_or_default();
    let mut top: Vec<PathBuf> = std::fs::read_dir(game_dir)
        .with_context(|| format!("reading {}", game_dir.display()))?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .collect();
    top.sort();
    let named = |path: &Path, ext: &str| {
        path.extension()
            .is_some_and(|e| e.eq_ignore_ascii_case(ext))
    };
    let exe = manifest_exes(&manifest)
        .into_iter()
        .map(|name| game_dir.join(name))
        .find(|path| path.is_file())
        .or_else(|| {
            let exes: Vec<&PathBuf> = top.iter().filter(|p| named(p, "exe")).collect();
            (exes.len() == 1).then(|| exes[0].clone())
        })
        .with_context(|| {
            format!(
                "no title executable in {}: its MicrosoftGame.config names none that is \
                 there, and the folder has no single .exe",
                game_dir.display()
            )
        })?;
    let archive = game_dir.join("shaders.rpf");
    Ok(archive
        .is_file()
        .then_some(archive)
        .into_iter()
        .chain([exe])
        .chain(top.into_iter().filter(|p| named(p, "dll")))
        .collect())
}

/// The executables a title's manifest names, in order: the `Name` of each
/// `<Executable …/>` (inside its `<ExecutableList>`).
fn manifest_exes(manifest: &str) -> Vec<String> {
    let mut names = Vec::new();
    let mut rest = manifest;
    while let Some(at) = rest.find("<Executable") {
        rest = &rest[at + "<Executable".len()..];
        // `<ExecutableList>` and the like are not an executable.
        if !rest.starts_with(char::is_whitespace) {
            continue;
        }
        let tag = &rest[..rest.find('>').unwrap_or(rest.len())];
        if let Some(name) = attribute(tag, "Name") {
            names.push(name.to_owned());
        }
    }
    names
}

/// The value of attribute `name` in a tag's text (quoted either way).
fn attribute<'a>(tag: &'a str, name: &str) -> Option<&'a str> {
    let mut rest = tag;
    while let Some(at) = rest.find(name) {
        let whole = rest[..at].ends_with(char::is_whitespace);
        rest = &rest[at + name.len()..];
        let Some(value) = rest.trim_start().strip_prefix('=') else {
            continue;
        };
        let value = value.trim_start();
        let Some(quote) = value.chars().next().filter(|q| *q == '"' || *q == '\'') else {
            continue;
        };
        if whole {
            let value = &value[1..];
            return value.find(quote).map(|end| &value[..end]);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The manifest's executables, not its list's tag nor other attributes
    /// ending in `Name`.
    #[test]
    fn the_manifest_names_its_executables() {
        let manifest = r#"<Game>
          <ExecutableList>
            <Executable Name="Title.exe" Id="Game" TargetDeviceFamily="Scarlett"/>
            <Executable OverrideDisplayName="x" Name = 'Other.exe' />
          </ExecutableList>
          <ShellVisuals DefaultDisplayName="Title"/>
        </Game>"#;
        assert_eq!(manifest_exes(manifest), ["Title.exe", "Other.exe"]);
        assert!(manifest_exes("").is_empty());
        assert!(manifest_exes("<ExecutableList></ExecutableList>").is_empty());
    }

    /// A folder (a temporary one, with empty files) gives its archive, the
    /// executable its manifest names and its DLLs; with no manifest, its one
    /// `.exe`; with two and no manifest, none.
    #[test]
    fn a_game_folder_gives_its_archive_executable_and_dlls() {
        let dir = std::env::temp_dir().join(format!("victory-title-files-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        for name in [
            "shaders.rpf",
            "Title.exe",
            "Crash.exe",
            "b.dll",
            "A.DLL",
            "data.bin",
        ] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
        std::fs::write(dir.join("sub").join("c.dll"), b"").unwrap();
        let config = r#"<Executable Name="Title.exe" Id="Game"/>"#;
        std::fs::write(dir.join("MicrosoftGame.config"), config).unwrap();
        let names = |files: Vec<PathBuf>| -> Vec<String> {
            files
                .iter()
                .map(|f| f.file_name().unwrap().to_string_lossy().into_owned())
                .collect()
        };
        let files = title_files(&dir).unwrap();
        assert_eq!(names(files), ["shaders.rpf", "Title.exe", "A.DLL", "b.dll"]);
        std::fs::remove_file(dir.join("MicrosoftGame.config")).unwrap();
        assert!(
            title_files(&dir).is_err(),
            "two executables and no manifest"
        );
        std::fs::remove_file(dir.join("Crash.exe")).unwrap();
        std::fs::remove_file(dir.join("shaders.rpf")).unwrap();
        assert_eq!(
            names(title_files(&dir).unwrap()),
            ["Title.exe", "A.DLL", "b.dll"]
        );
        std::fs::remove_dir_all(&dir).unwrap();
        assert!(title_files(&dir).is_err(), "no folder");
    }
}
