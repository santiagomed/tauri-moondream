use crate::{utils::load_image, Error, Generation, Token};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::{
    generation::LogitsProcessor,
    models::moondream::{Config, Model},
};
use tokenizers::Tokenizer;

fn build_model_and_tokenizer(
    api: &hf_hub::api::sync::Api,
    device: &Device,
) -> Result<(Model, Tokenizer), Error> {
    let model_id = "vikhyatk/moondream2".to_string();
    let repo = api.repo(hf_hub::Repo::new(model_id, hf_hub::RepoType::Model));
    let model_file = repo.get("model.safetensors")?;
    let tokenizer = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer)?;
    let config = Config::v2();

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], DType::F16, device)? };
    let model = Model::new(&config, vb)?;
    tracing::debug!("Model and tokenizer loaded");
    Ok((model, tokenizer))
}

fn get_image_embeddings(image: String, device: &Device) -> Result<Tensor, Error> {
    tracing::debug!("Loading image {}", image);
    let image = load_image(&image)?
        .to_dtype(DType::F16)?
        .to_device(device)?
        .unsqueeze(0)?;
    Ok(image)
}

pub fn build_pipeline(
    prompt: String,
    image: String,
    device: &Device,
    cache: &hf_hub::Cache,
) -> Result<Pipeline, Error> {
    let api = hf_hub::api::sync::ApiBuilder::from_cache(cache.clone()).build()?;
    let (model, tokenizer) = build_model_and_tokenizer(&api, device)?;
    let prompt = format!("\n\nQuestion: {}\nAnswer:", prompt);
    let tokens = tokenizer.encode(prompt, true)?;
    if tokens.is_empty() {
        return Err(Error::InputError("Prompt is empty".to_string()));
    }
    let tokens = tokens.get_ids().to_vec();
    let image_embeds = get_image_embeddings(image, device)?.apply(model.vision_encoder())?;
    tracing::debug!("Generated image embeddings: {:?}", image_embeds);
    Pipeline::new(model, tokenizer, device, &tokens, image_embeds)
}

pub struct PipelineIter<'a> {
    pipeline: &'a mut Pipeline,
    tokens: Vec<u32>,
    image_embeds: Tensor,
    generated_tokens: Vec<u32>,
    last: bool,
    i: usize,
}

pub struct Pipeline {
    model: Model,
    device: Device,
    tokenizer: Tokenizer,
    logits_processor: LogitsProcessor,
    tokens: Vec<u32>,
    image_embeds: Tensor,
    special_token: u32,
}

impl Pipeline {
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        device: &Device,
        tokens: &Vec<u32>,
        image_embeds: Tensor,
    ) -> Result<Self, Error> {
        let logits_processor = LogitsProcessor::new(0, None, None);
        // Moondream tokenizer bos_token and eos_token is "<|endoftext|>"
        // https://huggingface.co/vikhyatk/moondream2/blob/main/special_tokens_map.json
        let special_token = match tokenizer.get_vocab(true).get("<|endoftext|>") {
            Some(token) => *token,
            None => {
                return Err(Error::SpecialTokenNotFound(
                    "Special token not found".to_string(),
                ))
            }
        };
        Ok(Self {
            model,
            tokenizer,
            logits_processor,
            device: device.clone(),
            tokens: tokens.clone(),
            special_token,
            image_embeds,
        })
    }

    pub fn iter(&mut self) -> PipelineIter {
        PipelineIter {
            tokens: self.tokens.clone(),
            image_embeds: self.image_embeds.clone(),
            generated_tokens: vec![],
            pipeline: self,
            i: 0,
            last: false,
        }
    }
}

impl<'a> PipelineIter<'a> {
    fn inner_next(&mut self) -> Result<Generation, Error> {
        let special_token = self.pipeline.special_token;
        let input = Tensor::new(self.tokens.as_slice(), &self.pipeline.device)?.unsqueeze(0)?;
        let logits = if self.i > 0 {
            self.pipeline.model.text_model.forward(&input)?
        } else {
            let bos_token = Tensor::new(&[special_token], &self.pipeline.device)?.unsqueeze(0)?;
            let logits = self.pipeline.model.text_model.forward_with_img(
                &bos_token,
                &input,
                &self.image_embeds,
            )?;
            logits
        };
        let logits = logits.squeeze(0)?.to_dtype(DType::F16)?;
        let next_token = self.pipeline.logits_processor.sample(&logits)?;
        let text = self.pipeline.tokenizer.decode(&[next_token], true)?;
        tracing::debug!("Generated token: {}", text);
        self.generated_tokens.push(next_token);
        self.tokens = vec![next_token];
        let stop = next_token == special_token;
        let generated_text = if stop {
            tracing::debug!("End of text. Stopping...");
            Some(
                self.pipeline
                    .tokenizer
                    .decode(&self.generated_tokens, true)?,
            )
        } else {
            None
        };
        self.i += 1;
        Ok(Generation {
            token: Token {
                id: next_token as usize,
                text,
                special: stop,
            },
            generated_text,
            details: None,
        })
    }
}

impl<'a> Iterator for PipelineIter<'a> {
    type Item = Result<Generation, Error>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.last {
            return None;
        }
        let generation = self.inner_next();
        if let Ok(generation) = &generation {
            if generation.generated_text.is_some() {
                self.last = true;
            }
        }
        Some(generation)
    }
}
