use tauri::{
  plugin::{Builder, TauriPlugin},
  Manager, Runtime,
};

use std::{collections::HashMap, sync::Mutex};

pub use models::*;

#[cfg(desktop)]
mod desktop;
#[cfg(mobile)]
mod mobile;

mod commands;
mod error;
mod models;

pub use error::{Error, Result};

#[cfg(desktop)]
use desktop::Image;
#[cfg(mobile)]
use mobile::Image;

#[derive(Default)]
struct MyState(Mutex<HashMap<String, String>>);

/// Extensions to [`tauri::App`], [`tauri::AppHandle`] and [`tauri::Window`] to access the image APIs.
pub trait ImageExt<R: Runtime> {
  fn image(&self) -> &Image<R>;
}

impl<R: Runtime, T: Manager<R>> crate::ImageExt<R> for T {
  fn image(&self) -> &Image<R> {
    self.state::<Image<R>>().inner()
  }
}

/// Initializes the plugin.
pub fn init<R: Runtime>() -> TauriPlugin<R> {
  Builder::new("image")
    .invoke_handler(tauri::generate_handler![commands::execute])
    .setup(|app, api| {
      #[cfg(mobile)]
      let image = mobile::init(app, api)?;
      #[cfg(desktop)]
      let image = desktop::init(app, api)?;
      app.manage(image);

      // manage state so it is accessible by the commands
      app.manage(MyState::default());
      Ok(())
    })
    .build()
}
