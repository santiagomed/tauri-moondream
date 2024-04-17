import { invoke } from "@tauri-apps/api/core";
import { UnlistenFn, listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { info, error } from "@tauri-apps/plugin-log";
import { Payload } from "./types";

let errorMessage: HTMLParagraphElement | null;
let modelResponse: HTMLParagraphElement | null;
let prompt: HTMLInputElement | null;
let image: HTMLInputElement | null;
let imagePreview: HTMLImageElement | null;
let loading = false;
let isAborted = false;

async function getTextGenerationStream() {
  if (!modelResponse) {
    throw new Error("Model response element not found");
  }
  loading = true;
  modelResponse.textContent = "Loading image and model...";

  let unlisten: Promise<UnlistenFn> = Promise.resolve(() => {});
  const response = new ReadableStream({
    start(controller) {
      unlisten = listen("text-generation", (output: any) => {
        if (!output) {
          // If no output is received, consider closing the stream or logging an error
          info("Received empty output, possible end of data");
          controller.close();
          return;
        }
        info(`Received output: ${JSON.stringify(output)}`);
        const data = output.payload as Payload;
        controller.enqueue(data);
      });
    },
    cancel() {
      info("Stream cancelled");
    },
  });

  info("Invoking generate");

  await invoke("generate", {
    prompt: prompt && prompt.value,
    image: image && image.value,
  });

  info("Invoked generate");
  const reader = response.getReader();
  try {
    while (true) {
      info("Reading from reader");
      const { value, done } = await reader.read();
      if (loading) {
        modelResponse.textContent = "";
        loading = false;
      }

      if (done) {
        info("Stream completed, no more data");
        break;
      }

      info("Reader value", value);
      if (isAborted) {
        isAborted = false;
        await invoke("stop");
        break;
      }

      // final message
      if (value.generated_text) {
        break;
      }

      if (!value.token.special) {
        modelResponse.textContent += value.token.text;
      }
    }
  } catch (err) {
    error(`Error: ${err}`);
    errorMessage!.textContent = `Error: ${err}`;
  } finally {
    await unlisten;
    reader.releaseLock();
  }
}

async function sendInput() {
  try {
    isAborted = false;
    await getTextGenerationStream();
  } catch (err) {
    error(`Error: ${err}`);
    errorMessage!.textContent = `Error: ${err}`;
  }
}

async function openImage() {
  info("Opening image");
  let result;
  try {
    result = await open({
      directory: false,
      multiple: false,
      filter: [{ name: "Images", extensions: ["jpg", "png", "jpeg"] }],
    });
  } catch (err) {
    errorMessage!.textContent = `Error: ${err}`;
    return;
  }

  if (result && image) {
    try {
      const imagePath: string = await invoke("copy_image", {
        src: result.path,
      });
      // split iamge path to get the file name
      imagePreview!.src = `./assets/${imagePath.split("/").pop()}`;
      image.value = imagePath;
    } catch (err) {
      errorMessage!.textContent = `Error: ${err}`;
      return;
    }
  }
}

async function stop() {
  try {
    modelResponse!.textContent = "";
    await invoke("stop");
    isAborted = true;
  } catch (err) {
    errorMessage!.textContent = `Error: ${err}`;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  prompt = document.querySelector("#prompt-input");
  image = document.querySelector("#image-input");
  modelResponse = document.querySelector("#response");
  errorMessage = document.querySelector("#error-message");
  imagePreview = document.querySelector("#image-preview");

  document.querySelector("#image-upload")?.addEventListener("click", () => {
    openImage();
  });

  document.querySelector("#stop")?.addEventListener("click", () => {
    stop();
  });

  document.querySelector("#input-form")?.addEventListener("submit", (e) => {
    e.preventDefault();
    if (prompt && prompt.value && image && image.value) {
      sendInput();
    }
  });
});
