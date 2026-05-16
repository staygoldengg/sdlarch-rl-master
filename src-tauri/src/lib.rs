use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use tauri::{Manager, State};

// ─── Shared state: holds the spawned AI backend process ──────────────────────
struct ServerProcess(Mutex<Option<Child>>);

// ── Resolve striker-server.exe path ───────────────────────────────────────────
// Priority:
//   1. <own exe dir>/striker-server/striker-server.exe  (installed release)
//   2. <own exe dir>/striker-server.exe                 (flat release layout)
//   3. dist/striker-server/striker-server.exe           (post-PyInstaller dev)
//   4. dist/striker-server.exe                          (onefile dev)
fn find_server_exe(app: &tauri::AppHandle) -> Option<PathBuf> {
    // Tauri resource dir contains external binaries in release builds
    let resource_base = app.path().resource_dir().ok();

    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_owned()));

    let mut candidates: Vec<PathBuf> = vec![];

    if let Some(ref res) = resource_base {
        candidates.push(res.join("striker-server").join("striker-server.exe"));
        candidates.push(res.join("striker-server.exe"));
    }
    if let Some(ref dir) = exe_dir {
        candidates.push(dir.join("striker-server").join("striker-server.exe"));
        candidates.push(dir.join("striker-server.exe"));
    }
    // Dev fallback: relative to cwd (project root when running `npm run tauri dev`)
    candidates.push(PathBuf::from("dist/striker-server/striker-server.exe"));
    candidates.push(PathBuf::from("dist/striker-server.exe"));

    candidates.into_iter().find(|p| p.exists())
}

// ─────────────────────────────────────────────────────────────────────────────
// server_health — TCP-ping localhost:8000 to see if FastAPI is reachable
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn server_health() -> bool {
    let addr = "127.0.0.1:8000".parse().unwrap();
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

// ─────────────────────────────────────────────────────────────────────────────
// start_ai_server — spawn striker-server.exe (the bundled FastAPI backend)
// Returns: "started" | "already_running" | "not_found" | error string
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn start_ai_server(
    app: tauri::AppHandle,
    state: State<'_, ServerProcess>,
) -> Result<String, String> {
    let mut guard = state.0.lock().unwrap();
    if guard.is_some() {
        return Ok("already_running".to_string());
    }
    let addr = "127.0.0.1:8000".parse().unwrap();
    if TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
        return Ok("already_running".to_string());
    }

    let server_path = find_server_exe(&app).ok_or("striker-server.exe not found".to_string())?;

    let child = Command::new(&server_path)
        .spawn()
        .map_err(|e| format!("spawn failed: {e}"))?;

    *guard = Some(child);
    Ok("started".to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// stop_ai_server — kill the managed backend process
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn stop_ai_server(state: State<'_, ServerProcess>) -> String {
    let mut guard = state.0.lock().unwrap();
    if let Some(mut child) = guard.take() {
        let _ = child.kill();
        "stopped".to_string()
    } else {
        "not_managed".to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Legacy shim — kept so existing frontend calls don't break
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn start_capture_server(
    app: tauri::AppHandle,
    state: State<'_, ServerProcess>,
    _server_dir: String,
) -> Result<String, String> {
    start_ai_server(app, state)
}

#[tauri::command]
fn stop_capture_server(state: State<'_, ServerProcess>) -> String {
    stop_ai_server(state)
}

// ─────────────────────────────────────────────────────────────────────────────
// execute_macro — forward macro steps to the FastAPI /input/macro endpoint
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn execute_macro(steps: Vec<String>) -> Result<(), String> {
    use std::io::Write;
    let addr = "127.0.0.1:8000".parse().unwrap();
    if TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_err() {
        return Err("AI server offline — it will start automatically on next launch.".to_string());
    }
    // POST /input/macro with {"name": first_step}
    let macro_name = steps.first().map(|s| s.as_str()).unwrap_or("nlight");
    let body = format!(r#"{{"name":"{}"}}"#, macro_name);
    let mut stream = TcpStream::connect("127.0.0.1:8000").map_err(|e| e.to_string())?;
    let req = format!(
        "POST /input/macro HTTP/1.1\r\nHost: 127.0.0.1:8000\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
// Auto-start helper — spawns the server and polls until port 8000 is open
// (called in setup; non-blocking after spawn)
// ─────────────────────────────────────────────────────────────────────────────
fn auto_start_server(app: &tauri::AppHandle, state: &ServerProcess) {
    // Already running externally (e.g. start_server.ps1 was used)
    let addr = "127.0.0.1:8000".parse().unwrap();
    if TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok() {
        return;
    }
    let Some(server_path) = find_server_exe(app) else {
        // Server exe not found — user must start manually via start_server.ps1
        return;
    };
    let Ok(child) = Command::new(&server_path).spawn() else {
        return;
    };
    *state.0.lock().unwrap() = Some(child);
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let server_state = ServerProcess(Mutex::new(None));

    tauri::Builder::default()
        .manage(server_state)
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            // Auto-launch the AI backend when the window opens
            let state = app.state::<ServerProcess>();
            auto_start_server(app.handle(), &state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            server_health,
            start_ai_server,
            stop_ai_server,
            start_capture_server,
            stop_capture_server,
            execute_macro,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
