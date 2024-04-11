use candle::Device;
use serde::{Deserialize, Serialize};
use tauri::Manager;
use tauri_plugin_log::{Target, TargetKind};
use tracing::{error, info};

pub mod moondream;
pub mod utils;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    Json(#[from] serde_json::Error),

    #[error(transparent)]
    Tauri(#[from] tauri::Error),

    #[error(transparent)]
    Lock(#[from] tokio::sync::TryLockError),

    #[error(transparent)]
    Api(#[from] hf_hub::api::sync::ApiError),

    #[error(transparent)]
    Candle(#[from] candle::Error),

    #[error(transparent)]
    Tokenizer(#[from] Box<dyn std::error::Error + Send + Sync>),

    #[error("Model {0} was not found")]
    ModelNotFound(String),

    #[error("Special token {0} was not found")]
    SpecialTokenNotFound(String),

    #[error("Input error {0}")]
    InputError(String),
}

impl Serialize for Error {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::ser::Serializer,
    {
        serializer.serialize_str(self.to_string().as_ref())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Token {
    id: usize,
    text: String,
    special: bool,
}
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Generation {
    token: Token,
    generated_text: Option<String>,
    details: Option<bool>,
}

#[tauri::command]
async fn stop(state: tauri::State<'_, State>) -> Result<(), Error> {
    tracing::info!("STOP");
    let mut tx = state.tx.try_lock()?;
    let tmptx = (*tx).take();
    if let Some(tx) = tmptx {
        if let Err(_) = tx.send(()) {
            error!("Could not send stop signal");
        }
    }
    Ok(())
}

#[tauri::command]
async fn generate(
    app: tauri::AppHandle,
    state: tauri::State<'_, State>,
    prompt: String,
    image: String,
) -> Result<(), Error> {
    tracing::debug!("Generating for {prompt} and {image}");
    let (newtx, mut rx) = tokio::sync::oneshot::channel();
    let cache = state.cache.clone();
    let device = state.device.clone();
    tokio::task::spawn_blocking(move || {
        let mut moondream = moondream::build_pipeline(prompt, image, &device, &cache)?;
        info!("Pipeline created");
        for generation in moondream.iter() {
            let generation = generation?;
            app.emit("text-generation", generation)?;
            if let Ok(_) = rx.try_recv() {
                break;
            }
        }
        Ok::<(), Error>(())
    });
    let mut tx = state.tx.try_lock()?;
    let tmptx = (*tx).take();
    if let Some(tx) = tmptx {
        if let Err(_) = tx.send(()) {
            error!("Could not send to tx");
        }
    }
    *tx = Some(newtx);
    Ok(())
}

#[allow(unused_variables)]
fn cache(path: &std::path::Path) -> hf_hub::Cache {
    #[cfg(not(mobile))]
    let cache = hf_hub::Cache::default();
    #[cfg(mobile)]
    let cache = {
        std::fs::create_dir_all(path).expect("Could not create dir");
        let cache = hf_hub::Cache::new(path.to_path_buf());
        let token_path = cache.token_path();
        cache
    };
    cache
}

struct State {
    cache: hf_hub::Cache,
    device: Device,
    tx: tokio::sync::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(
            tauri_plugin_log::Builder::new()
                .targets([
                    Target::new(TargetKind::Stdout),
                    Target::new(TargetKind::LogDir { file_name: None }),
                    Target::new(TargetKind::Webview),
                ])
                .build(),
        )
        .plugin(tauri_plugin_shell::init())
        .invoke_handler(tauri::generate_handler![generate, stop])
        .setup(move |app| {
            info!("Start the run");
            info!(
                "avx: {}, neon: {}, simd128: {}, f16c: {}",
                candle::utils::with_avx(),
                candle::utils::with_neon(),
                candle::utils::with_simd128(),
                candle::utils::with_f16c()
            );
            let path = app.path().local_data_dir().expect("Have a local data dir");
            let cache = cache(&path);
            tracing::info!("get the device");
            let device = if candle::utils::cuda_is_available() {
                Device::new_cuda(0)?
            // Simulator doesn't support MPS (Metal Performance Shader).
            } else if candle::utils::metal_is_available()
                && std::env::var("TARGET").expect("TARGET env var not found")
                    != "aarch64-apple-ios-sim"
            {
                Device::new_metal(0)?
            } else {
                Device::Cpu
            };
            tracing::info!("device: {:?}", device);
            app.manage(State {
                cache,
                device,
                tx: tokio::sync::Mutex::new(None),
            });
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
