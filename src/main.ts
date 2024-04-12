import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { Payload } from "./types";

let modelResponse: HTMLParagraphElement | null;
let prompt: HTMLInputElement | null;
let image: HTMLInputElement | null;
let isAborted = false;

async function getTextGenerationStream() {
  if (!modelResponse) {
    throw new Error("Model response element not found");
  }

  let unlisten = undefined;
  const response = new ReadableStream({
    start(controller) {
      unlisten = listen("text-generation", (output: any) => {
        output = output.payload as Payload;
        controller.enqueue(output);
      });
    },
  });

  console.log("Invoking generate");

  await invoke("generate", {
    prompt: prompt && prompt.value,
    image: image && image.value,
  });

  console.log("Invoked generate");
  let done = false;
  const reader = response.getReader();
  while (!done) {
    const { value } = await reader.read();
    const output = value;

    console.log("Output", output);

    if (!output) {
      break;
    }

    if (isAborted) {
      isAborted = false;
      console.log("STOP");
      await invoke("stop");
      break;
    }

    // final message
    if (output.generated_text) {
      modelResponse.textContent = output.generated_text;
      break;
    }

    if (!output.token.special) {
      modelResponse.textContent += output.token.text;
    }
  }
  await unlisten;
}

async function sendInput() {
  try {
    isAborted = false;
    await getTextGenerationStream();
  } catch (err) {
    console.error(err);
  }
}

async function openImage() {
  console.log("Opening image");
  let result;
  try {
    result = await open({
      directory: false,
      multiple: false,
      filter: [{ name: "Images", extensions: ["jpg", "png", "jpeg"] }],
    });
  } catch (err) {
    modelResponse!.textContent = `Error: ${err}`;
    return;
  }

  console.log(result);

  if (result && image) {
    image.value = result.path;
  }
}

window.addEventListener("DOMContentLoaded", () => {
  prompt = document.querySelector("#prompt-input");
  image = document.querySelector("#image-input");
  modelResponse = document.querySelector("#response");

  document.querySelector("#image-upload")?.addEventListener("click", (e) => {
    openImage();
  });

  document.querySelector("#input-form")?.addEventListener("submit", (e) => {
    e.preventDefault();
    if (prompt && prompt.value && image && image.value) {
      sendInput();
    }
  });
});
