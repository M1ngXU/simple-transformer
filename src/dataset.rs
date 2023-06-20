use std::collections::HashMap;
use std::fs::File;

use bincode::Options;
use dfdx::data::ExactSizeDataset;
use dfdx::tensor::{Cpu, Tensor, TensorFromVec, Tensorlike};
use flate2::read::ZlibDecoder;
use itertools::*;
use rust_tokenizers::tokenizer::{Gpt2Tokenizer, Tokenizer, TruncationStrategy};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::get_cpu;

pub type Label = bool;
pub type Token = usize;
pub type Gpt2Token = usize;
pub type Tokens = Tensor<(usize,), Token, Cpu>;
pub const AMOUNT_STARS: usize = 1;
pub const MAX_LEN: usize = 500;

static TRUNCATION_STRATEGY: TruncationStrategy = TruncationStrategy::DoNotTruncate;
fn get_tokenizer() -> Gpt2Tokenizer {
    Gpt2Tokenizer::from_file("vocab.json", "merges.txt", false).unwrap()
}
#[derive(Deserialize)]
pub struct Record {
    pub label: String,
    pub title: String,
    pub content: String,
}

#[derive(Serialize, Deserialize)]
pub struct Data {
    pub training_data: Dataset,
    pub validation_data: Dataset,
    pub test_data: Dataset,
    #[serde(skip, default = "get_tokenizer")]
    pub tokenizer: Gpt2Tokenizer,
    vocabulary: HashMap<Gpt2Token, Token>,
}
#[derive(Deserialize, Serialize)]
pub struct Dataset {
    data: Vec<Entry>,
}
impl ExactSizeDataset for Dataset {
    type Item<'a> = &'a Entry;

    fn get(&self, index: usize) -> Self::Item<'_> {
        &self.data[index]
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

#[derive(Deserialize, Serialize, Clone)]
pub struct Entry {
    #[serde(serialize_with = "serialize_tokens")]
    #[serde(deserialize_with = "deserialize_tokens")]
    pub tokens: Tokens,
    pub label: Label,
}
fn serialize_tokens<S: Serializer>(tokens: &Tokens, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.collect_seq(tokens.data().unwrap().iter().copied())
}
fn deserialize_tokens<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Tokens, D::Error> {
    let tokens = Vec::<usize>::deserialize(deserializer).unwrap();
    let shape = (tokens.len(),);
    Ok(get_cpu().tensor_from_vec(tokens, shape))
}

#[cfg(feature = "save-data")]
fn save_training_data() -> Data {
    use crate::model::VOCAB;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use rayon::prelude::{
        IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelBridge, ParallelIterator,
    };
    let tokenizer = get_tokenizer();
    let data = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("train.csv")
        .unwrap()
        .deserialize::<Record>()
        //.take(10)
        .par_bridge()
        .filter_map(|r| {
            let record = r.unwrap();
            let title = record.content;
            (1..=MAX_LEN)
                .contains(&title.split_whitespace().count())
                .then_some((title, record.label == "2"))
        })
        .collect::<Vec<_>>();

    let mut data_by_stars = data
        .par_iter()
        .map(|(text, positive)| {
            (
                *positive,
                tokenizer
                    .encode(&text, None, usize::MAX, &TRUNCATION_STRATEGY, 1)
                    .token_ids
                    .into_iter()
                    .map(|t| t as Gpt2Token)
                    .collect_vec(),
            )
        })
        .collect::<Vec<_>>()
        .into_iter()
        .into_group_map();

    let smallest_group_size = data_by_stars
        .par_iter()
        .map(|(_, d)| d.len())
        .min()
        .unwrap_or(0);

    // same amount of labels
    for (_, data) in &mut data_by_stars {
        data.truncate(smallest_group_size);
    }

    let vocabulary: HashMap<_, _> = data_by_stars
        .values()
        .flatten()
        .flatten()
        .copied()
        .filter(|t| t > &0)
        .counts()
        .into_iter()
        .sorted_by_key(|(_, f)| *f)
        .rev()
        .take(VOCAB - 1) // 0 (unknown) always included
        .enumerate()
        .map(|(i, (token, _))| (token, i as Token + 1))
        .collect();

    for (_, data) in &mut data_by_stars {
        for tokens in data {
            for token in tokens {
                *token = *vocabulary.get(&token).unwrap_or(&0);
            }
        }
    }

    let training_data_ratio = 0.99;
    let validation_set_ratio = 0.01;

    assert_eq!(training_data_ratio + validation_set_ratio, 1.0);

    let training_data_size = (training_data_ratio * smallest_group_size as f32) as usize;
    let validation_data_size = (validation_set_ratio * smallest_group_size as f32) as usize;
    let dev = get_cpu();
    let training_data = data_by_stars
        .par_iter_mut()
        .flat_map_iter(|(label, data)| {
            data.drain(..training_data_size).map(|tokens| {
                let shape = (tokens.len(),);
                Entry {
                    tokens: dev.tensor_from_vec(tokens, shape),
                    label: *label,
                }
            })
        })
        .collect::<Vec<_>>();
    let validation_data = data_by_stars
        .par_iter_mut()
        .flat_map_iter(|(label, data)| {
            data.drain(..validation_data_size).map(|tokens| {
                let shape = (tokens.len(),);
                Entry {
                    tokens: dev.tensor_from_vec(tokens, shape),
                    label: *label,
                }
            })
        })
        .collect::<Vec<_>>();
    let test_data = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_path("test.csv")
        .unwrap()
        .deserialize::<Record>()
        .par_bridge()
        .filter_map(|r| {
            r.ok().map(|record| {
                let tokens = tokenizer
                    .encode(&record.content, None, usize::MAX, &TRUNCATION_STRATEGY, 1)
                    .token_ids
                    .into_iter()
                    .map(|t| *vocabulary.get(&(t as Gpt2Token)).unwrap_or(&0))
                    .collect::<Vec<_>>();
                let shape = (tokens.len(),);
                Entry {
                    tokens: dev.tensor_from_vec(tokens, shape),
                    label: record.label == "2",
                }
            })
        })
        .collect::<Vec<_>>();

    let group_sizes = data.into_iter().map(|(_, s)| s).counts();
    println!(
        "Smallest group: {}",
        group_sizes
            .iter()
            .find(|(_, d)| d == &&smallest_group_size)
            .unwrap()
            .0
    );
    println!("Smallest group size: {smallest_group_size}");
    println!("Group sizes: {group_sizes:?}");
    println!("Training data size: {}", training_data.len());
    println!("Validation data size: {}", validation_data.len());
    println!("Test data size: {}", test_data.len());

    let data = Data {
        training_data: Dataset {
            data: training_data,
        },
        validation_data: Dataset {
            data: validation_data,
        },
        test_data: Dataset { data: test_data },
        tokenizer,
        vocabulary,
    };

    bincode::DefaultOptions::new()
        .serialize_into(
            ZlibEncoder::new(File::create("data.bin").unwrap(), Compression::default()),
            &data,
        )
        .unwrap();

    println!("Saved!");

    data
}

impl Data {
    pub fn new() -> Self {
        #[cfg(feature = "save-data")]
        save_training_data();

        let data: Data = bincode::DefaultOptions::new()
            .deserialize_from(ZlibDecoder::new(File::open("data.bin").unwrap()))
            .unwrap();

        data
    }

    pub fn tokenize(&self, s: &str) -> Vec<Token> {
        self.tokenizer
            .encode(&s, None, usize::MAX, &TRUNCATION_STRATEGY, 1)
            .token_ids
            .into_iter()
            .map(|t| *self.vocabulary.get(&(t as Gpt2Token)).unwrap_or(&0))
            .collect_vec()
    }
}
