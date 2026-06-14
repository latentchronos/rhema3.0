use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread::JoinHandle;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::State;

const DEFAULT_OBS_PORT: u16 = 4763;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ObsOverlayPayload {
    pub output_id: String,
    pub theme: serde_json::Value,
    pub verse: serde_json::Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ObsOverlayStatus {
    pub active: bool,
    pub port: Option<u16>,
    pub main_url: Option<String>,
    pub alt_url: Option<String>,
    pub client_count: usize,
}

pub struct ObsOverlayServer {
    port: Option<u16>,
    stop: Arc<AtomicBool>,
    clients: Arc<Mutex<Vec<mpsc::Sender<String>>>>,
    latest_payload: Arc<Mutex<Option<String>>>,
    handle: Option<JoinHandle<()>>,
}

impl Default for ObsOverlayServer {
    fn default() -> Self {
        Self {
            port: None,
            stop: Arc::new(AtomicBool::new(false)),
            clients: Arc::new(Mutex::new(Vec::new())),
            latest_payload: Arc::new(Mutex::new(None)),
            handle: None,
        }
    }
}

impl ObsOverlayServer {
    /// Whether the overlay HTTP/SSE server is currently running (Bullet 4.4
    /// device-health probe).
    pub fn is_running(&self) -> bool {
        self.handle.is_some()
    }

    fn status(&self) -> ObsOverlayStatus {
        let client_count = self.clients.lock().map(|c| c.len()).unwrap_or_default();
        ObsOverlayStatus {
            active: self.handle.is_some(),
            port: self.port,
            main_url: self.port.map(|port| overlay_url(port, "main")),
            alt_url: self.port.map(|port| overlay_url(port, "alt")),
            client_count,
        }
    }

    fn start(&mut self, requested_port: Option<u16>) -> Result<ObsOverlayStatus, String> {
        if self.handle.is_some() {
            return Ok(self.status());
        }

        let port = requested_port.unwrap_or(DEFAULT_OBS_PORT);
        let listener = TcpListener::bind(("127.0.0.1", port))
            .map_err(|e| format!("failed to bind OBS overlay server on port {port}: {e}"))?;
        listener
            .set_nonblocking(true)
            .map_err(|e| format!("failed to configure OBS overlay server: {e}"))?;

        self.stop.store(false, Ordering::SeqCst);
        let stop = self.stop.clone();
        let clients = self.clients.clone();
        let latest_payload = self.latest_payload.clone();

        self.port = Some(port);
        self.handle = Some(std::thread::spawn(move || {
            while !stop.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let clients = clients.clone();
                        let latest_payload = latest_payload.clone();
                        std::thread::spawn(move || {
                            handle_client(stream, clients, latest_payload);
                        });
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(50));
                    }
                    Err(e) => {
                        log::warn!("[OBS] overlay accept failed: {e}");
                        std::thread::sleep(Duration::from_millis(250));
                    }
                }
            }
        }));

        Ok(self.status())
    }

    fn stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(port) = self.port {
            let _ = TcpStream::connect(("127.0.0.1", port));
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        self.port = None;
        if let Ok(mut clients) = self.clients.lock() {
            clients.clear();
        }
    }

    fn push(&mut self, payload: ObsOverlayPayload) -> Result<(), String> {
        let event_payload = serde_json::to_string(&payload)
            .map_err(|e| format!("failed to serialize OBS overlay payload: {e}"))?;
        if let Ok(mut latest) = self.latest_payload.lock() {
            *latest = Some(event_payload.clone());
        }

        let event = sse_event(&event_payload);
        if let Ok(mut clients) = self.clients.lock() {
            clients.retain(|tx| tx.send(event.clone()).is_ok());
        }
        Ok(())
    }
}

#[tauri::command]
pub fn start_obs_overlay(
    state: State<'_, Mutex<ObsOverlayServer>>,
    port: Option<u16>,
) -> Result<ObsOverlayStatus, String> {
    state.lock().map_err(|e| e.to_string())?.start(port)
}

#[tauri::command]
pub fn stop_obs_overlay(state: State<'_, Mutex<ObsOverlayServer>>) -> Result<(), String> {
    state.lock().map_err(|e| e.to_string())?.stop();
    Ok(())
}

#[tauri::command]
pub fn get_obs_overlay_status(
    state: State<'_, Mutex<ObsOverlayServer>>,
) -> Result<ObsOverlayStatus, String> {
    Ok(state.lock().map_err(|e| e.to_string())?.status())
}

#[tauri::command]
pub fn push_obs_overlay(
    state: State<'_, Mutex<ObsOverlayServer>>,
    payload: ObsOverlayPayload,
) -> Result<(), String> {
    state.lock().map_err(|e| e.to_string())?.push(payload)
}

fn overlay_url(port: u16, output_id: &str) -> String {
    format!("http://127.0.0.1:{port}/?output={output_id}")
}

fn handle_client(
    mut stream: TcpStream,
    clients: Arc<Mutex<Vec<mpsc::Sender<String>>>>,
    latest_payload: Arc<Mutex<Option<String>>>,
) {
    let mut buffer = [0_u8; 2048];
    let read = stream.read(&mut buffer).unwrap_or_default();
    let request = String::from_utf8_lossy(&buffer[..read]);
    let path = request
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .unwrap_or("/");

    if path.starts_with("/events") {
        serve_events(stream, clients, latest_payload);
    } else {
        serve_html(stream);
    }
}

fn serve_html(mut stream: TcpStream) {
    let body = OBS_OVERLAY_HTML;
    let response = format!(
        "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\n\r\n{}",
        body.len(),
        body
    );
    let _ = stream.write_all(response.as_bytes());
}

fn serve_events(
    mut stream: TcpStream,
    clients: Arc<Mutex<Vec<mpsc::Sender<String>>>>,
    latest_payload: Arc<Mutex<Option<String>>>,
) {
    let headers = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: keep-alive\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    if stream.write_all(headers.as_bytes()).is_err() {
        return;
    }

    if let Ok(latest) = latest_payload.lock() {
        if let Some(payload) = latest.as_ref() {
            let _ = stream.write_all(sse_event(payload).as_bytes());
        }
    }

    let (tx, rx) = mpsc::channel::<String>();
    if let Ok(mut clients) = clients.lock() {
        clients.push(tx);
    }

    while let Ok(event) = rx.recv() {
        if stream.write_all(event.as_bytes()).is_err() {
            break;
        }
        let _ = stream.flush();
    }
}

fn sse_event(payload: &str) -> String {
    format!("event: overlay\ndata: {payload}\n\n")
}

const OBS_OVERLAY_HTML: &str = r#"<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rhema OBS Overlay</title>
  <style>
    html, body {
      margin: 0;
      width: 100%;
      height: 100%;
      overflow: hidden;
      background: transparent;
      font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    #overlay {
      box-sizing: border-box;
      position: absolute;
      left: 8%;
      right: 8%;
      bottom: 8%;
      opacity: 0;
      transition: opacity 160ms ease-out;
      color: white;
      text-shadow: 0 2px 10px rgba(0,0,0,.65);
    }
    #overlay.visible { opacity: 1; }
    #reference {
      display: inline-block;
      margin-bottom: 14px;
      padding: 7px 12px;
      border-radius: 6px;
      background: rgba(0, 0, 0, .62);
      font-size: clamp(18px, 2.3vw, 36px);
      font-weight: 700;
      letter-spacing: .02em;
    }
    #text {
      display: block;
      padding: 18px 22px;
      border-radius: 8px;
      background: rgba(0, 0, 0, .58);
      font-size: clamp(28px, 4vw, 64px);
      font-weight: 650;
      line-height: 1.15;
    }
  </style>
</head>
<body>
  <main id="overlay" aria-live="polite">
    <div id="reference"></div>
    <div id="text"></div>
  </main>
  <script>
    const params = new URLSearchParams(location.search);
    const outputId = params.get("output") || "main";
    const overlay = document.getElementById("overlay");
    const reference = document.getElementById("reference");
    const text = document.getElementById("text");

    function render(payload) {
      if (payload.outputId !== outputId) return;
      const verse = payload.verse;
      if (!verse || !verse.reference || !Array.isArray(verse.segments)) {
        overlay.classList.remove("visible");
        reference.textContent = "";
        text.textContent = "";
        return;
      }
      reference.textContent = verse.reference;
      text.textContent = verse.segments
        .map((segment) => segment.verseNumber ? segment.verseNumber + " " + segment.text : segment.text)
        .join(" ");
      overlay.classList.add("visible");
    }

    const events = new EventSource("/events");
    events.addEventListener("overlay", (event) => {
      try { render(JSON.parse(event.data)); } catch (_) {}
    });
  </script>
</body>
</html>"#;
