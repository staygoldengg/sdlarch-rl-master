use std::net::TcpStream;
use std::process::{Child, Command};
use std::sync::Mutex;
use std::time::Duration;
use tauri::State;

// ─── Shared state: holds the spawned capture_server.py process ───────────────
struct ServerProcess(Mutex<Option<Child>>);

// ─────────────────────────────────────────────────────────────────────────────
// server_health — TCP-ping localhost:5000 to see if Flask is reachable
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn server_health() -> bool {
    let addr = "127.0.0.1:5000".parse().unwrap();
    TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_ok()
}

// ─────────────────────────────────────────────────────────────────────────────
// start_capture_server — spawn `python capture_server.py` in server_dir
// Returns: "started" | "already_running" | error string
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn start_capture_server(
    state: State<'_, ServerProcess>,
    server_dir: String,
) -> Result<String, String> {
    // If already managed or externally running, skip re-launch
    let mut guard = state.0.lock().unwrap();
    if guard.is_some() {
        return Ok("already_running".to_string());
    }
    let addr = "127.0.0.1:5000".parse().unwrap();
    if TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok() {
        return Ok("already_running".to_string());
    }

    let script = format!("{}\\capture_server.py", server_dir.trim_end_matches('\\'));
    let child = Command::new("python")
        .arg(&script)
        .current_dir(&server_dir)
        // Suppress console window in release builds
        .spawn()
        .map_err(|e| format!("spawn failed: {e}"))?;

    *guard = Some(child);
    Ok("started".to_string())
}

// ─────────────────────────────────────────────────────────────────────────────
// stop_capture_server — kill the managed process (or no-op if not managed)
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn stop_capture_server(state: State<'_, ServerProcess>) -> String {
    let mut guard = state.0.lock().unwrap();
    if let Some(mut child) = guard.take() {
        let _ = child.kill();
        "stopped".to_string()
    } else {
        "not_managed".to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// execute_macro — forward step strings to the Flask /api/macro/run endpoint
// (kept as a Tauri command so the existing macro runner in App.tsx still works)
// ─────────────────────────────────────────────────────────────────────────────
#[tauri::command]
fn execute_macro(steps: Vec<String>) -> Result<(), String> {
    // Build the steps_text the Flask endpoint expects
    let steps_text = steps.join("\n");
    let body = format!(
        r#"{{"steps":"{}","loops":1,"delay_ms":0}}"#,
        steps_text.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n")
    );
    let addr = "127.0.0.1:5000".parse().unwrap();
    if TcpStream::connect_timeout(&addr, Duration::from_millis(300)).is_err() {
        return Err("Flask server offline — start it in the Fight Engine tab first.".to_string());
    }
    // Fire-and-forget via a raw HTTP/1.1 POST over std TCP (no extra crates needed)
    use std::io::Write;
    let mut stream = TcpStream::connect("127.0.0.1:5000")
        .map_err(|e| e.to_string())?;
    let req = format!(
        "POST /api/macro/run HTTP/1.1\r\nHost: 127.0.0.1:5000\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(), body
    );
    stream.write_all(req.as_bytes()).map_err(|e| e.to_string())?;
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(ServerProcess(Mutex::new(None)))
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            server_health,
            start_capture_server,
            stop_capture_server,
            execute_macro,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
