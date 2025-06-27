import {
  AutoTokenizer,
  AutoModelForCausalLM,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "./transformers.min.js";

const llm_model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct";
const cacheConfig = {
  cacheDir: "/tmp/hf-cache",
  localFilesOnly: false
};
let tokenizer = null;
let llm = null;

const LOAD_TIMEOUT = 300000; // 5 minutes timeout

async function loadLLMModel() {
  return new Promise(async (resolve, reject) => {
    const timeoutId = setTimeout(() => {
      self.postMessage({
        type: "status",
        status: "error",
        message: "Model loading timed out after 5 minutes"
      });
      reject(new Error("Model loading timed out"));
    }, LOAD_TIMEOUT);

    try {
      console.log("Starting LLM model download...");
      self.postMessage({
        type: "status",
        status: "downloading",
        message: "Starting model download..."
      });

      // Debug model files that will be downloaded
      const modelFiles = [
        "config.json",
        "tokenizer.json",
        "model.onnx"
      ];
      console.log("Expected model files:", modelFiles);

      console.log("Initiating tokenizer download from Hugging Face...");
      // Load tokenizer with explicit config
      tokenizer = await AutoTokenizer.from_pretrained(llm_model_id, {
        // Force fresh download and show URLs
        local_files_only: false,
        force_download: true,
        use_cache: false,
        use_browser_download: true,
        progress_callback: (x) => {
          console.log(`Download progress: ${x.file} - ${Math.round(x.progress*100)}%`);
          self.postMessage({ 
            type: "progress", 
            status: x.status,
            file: x.file,
            progress: x.progress,
            message: `Downloading ${x.file} (${Math.round(x.progress*100)}%)`,
            url: `https://huggingface.co/${llm_model_id}/resolve/main/${x.file}`
          });
        },
      });

      console.log("Initiating model download from Hugging Face...");
      // Load model with explicit config
      llm = await AutoModelForCausalLM.from_pretrained(llm_model_id, {
        // Force fresh download and show URLs
        local_files_only: false,
        force_download: true,
        use_cache: false,
        use_browser_download: true,
        dtype: "q8",
        device: "webgpu",
        progress_callback: (x) => {
          console.log(`Model download progress: ${x.file} - ${Math.round(x.progress*100)}%`);
          self.postMessage({
            type: "progress",
            status: x.status,
            file: x.file,
            progress: x.progress,
            message: `Downloading model file: ${x.file}`,
            url: `https://huggingface.co/${llm_model_id}/resolve/main/${x.file}`
          });
        },
      });

      // Verify model loaded properly
      if (!llm || !tokenizer) {
        throw new Error("Model failed to initialize");
      }
      
      // Test model with sample inference
      try {
        const testInput = tokenizer("Test", { return_tensors: "pt" });
        await llm.generate(testInput, { max_new_tokens: 1 });
        console.log("Model verification test passed");
      } catch (e) {
        throw new Error(`Model verification failed: ${e.message}`);
      }

      console.log("LLM model and tokenizer loaded successfully!");
      console.log("Model details:", {
        model: llm_model_id,
        dtype: "q8",
        device: "webgpu",
        parameters: "1.7B",
        verified: true
      });
      self.postMessage({ 
        type: "status",
        status: "ready",
        message: "Model loaded successfully!",
        details: {
          model: llm_model_id,
          size: "1.7B parameters",
          quantization: "8-bit"
        }
      });
      resolve();
    } catch (error) {
      console.error("Failed to load LLM model:", error);
      self.postMessage({ 
        type: "status",
        status: "error",
        message: `Error loading model: ${error.message}`
      });
      reject(error);
    } finally {
      clearTimeout(timeoutId);
    }
  });
}

// Simple test to verify script execution
console.log("Worker script loaded successfully - starting model download");
self.postMessage({
  type: "status",
  status: "init",
  message: "Worker initialized"
});

// Call the function to start loading the model when the script executes
loadLLMModel().catch(e => {
  console.error("Model loading failed completely:", e);
  self.postMessage({
    type: "status",
    status: "critical_error",
    message: `Failed to load model: ${e.message}`
  });
});

self.onmessage = async (event) => {
  const { type, prompt, context } = event.data;

  if (type === "generate") {
    try {
      if (!llm || !tokenizer) {
        throw new Error("LLM model not loaded. Please wait or check console for errors.");
      }

      const systemPrompt = "You are a helpful assistant that answers questions based on the provided context.";
      const fullPrompt = `${systemPrompt}\n\nContext: ${context}\n\nQuestion: ${prompt}\n\nAnswer:`;

      const inputs = tokenizer(fullPrompt, { return_tensors: "pt" });
      const { generated_ids } = await llm.generate(inputs, { max_new_tokens: 150 });
      const answer = tokenizer.decode(generated_ids[0], { skip_special_tokens: true }).split('Answer:')[1].trim();
      self.postMessage({ type: "result", answer: answer });
    } catch (error) {
      console.error("Failed to generate:", error);
      self.postMessage({ type: "status", status: "error", message: `Error generating: ${error.message}` });
    }
  }
};
