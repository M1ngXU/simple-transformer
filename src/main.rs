#![feature(thread_spawn_unchecked)]

use rayon::prelude::{ParallelBridge, ParallelIterator};
use rust_tokenizers::tokenizer::Tokenizer;
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Barrier, OnceLock, RwLock};
use std::thread::available_parallelism;
use std::time::{Duration, Instant};

use dfdx::data::ExactSizeDataset;
use dfdx::optim::{Adam, AdamConfig};
use dfdx::prelude::*;
use itertools::Itertools;
use model::{SimpleTransformer, SimpleTransformerBuilt};
use rand::thread_rng;

use crate::dataset::{Data, Entry};

mod dataset;
mod lsh;
mod model;

const BATCH_SIZE: usize = 128;
const VALIDATION_BATCH_SIZE: usize = BATCH_SIZE;
const EPOCHS: usize = 10000;
const DATA_PER_BATCH: usize = 32;
const NO_LOSS_IMPROVEMENT: usize = 5;

/// Amount of digits required to display [`EPOCHS`] as decimal.
const EPOCH_SIZE: usize = (EPOCHS - 1).ilog10() as usize + 1;
/// Amount of digits required to display [`DATA_PER_BATCH`] as decimal.
const DATA_PER_BATCH_SIZE: usize = (DATA_PER_BATCH - 1).ilog10() as usize + 1;

static PARALLELISM: OnceLock<usize> = OnceLock::new();
fn get_parallelism() -> usize {
	*PARALLELISM.get_or_init(|| available_parallelism().unwrap().get())
}
static CPU: OnceLock<Cpu> = OnceLock::new();
pub fn get_cpu() -> Cpu {
	CPU.get_or_init(|| Cpu::default()).clone()
}

#[allow(dead_code)]
fn argmax(arr: impl IntoIterator<Item = f32>) -> usize {
	arr.into_iter()
		.enumerate()
		.max_by(|(_, a), (_, b)| a.total_cmp(b))
		.unwrap_or_default()
		.0
}

fn thread_fn(
	model: Arc<RwLock<SimpleTransformerBuilt>>,
	optimizer: &RwLock<Adam<SimpleTransformerBuilt, f32, Cpu>>,
	data: Arc<Vec<Entry>>,
	data_index: Arc<AtomicUsize>,
	data_left: Arc<AtomicUsize>,
	report_done: Sender<(f32, usize)>,
	barrier_forward: Arc<Barrier>,
	barrier_new_batch: Arc<Barrier>,
) {
	let dev = get_cpu();

	let read_lock = model.read().unwrap();
	let mut gradients = read_lock.alloc_grads();

	drop(read_lock);
	loop {
		let mut avg_loss = 0.0;
		let mut correct = 0;

		barrier_forward.wait();

		// if `data_left` overflows, it is usize::MAX; `fetch_sub` returns the last
		// value => valid range is 1..=BATCH_SIZE
		while (1..=BATCH_SIZE).contains(&data_left.fetch_sub(1, Ordering::SeqCst)) {
			let mut index = data_index.load(Ordering::SeqCst);
			index = (index + 1) % data.len();
			data_index.store(index, Ordering::SeqCst);
			let Entry { tokens, label } = data[index].clone();
			let read_lock = model.read().unwrap();
			let output = read_lock.forward(tokens.traced(gradients));
			drop(read_lock);
			// let target = dev
			// 	.one_hot_encode(Const::<AMOUNT_STARS>, [stars])
			// 	.reshape::<Rank1<AMOUNT_STARS>>();
			// let argmax = argmax(output.array());
			// if argmax == stars {
			// 	correct += 1;
			// }
			// let loss = cross_entropy_with_logits_loss(output, target) / BATCH_SIZE as
			// f32;
			let pred = output.array()[0];
			if pred.is_sign_positive() && label || pred.is_sign_negative() && !label {
				correct += 1;
			}
			let loss = binary_cross_entropy_with_logits_loss(
				output,
				dev.tensor([if label { 1.0 } else { 0.0 }]),
			) / BATCH_SIZE as f32;
			avg_loss += loss.array();
			gradients = loss.backward();
		}
		barrier_new_batch.wait();
		let mut model_lock = model.write().unwrap();
		let mut optimizer_lock = optimizer.write().unwrap();
		optimizer_lock.update(&mut model_lock, &gradients).unwrap();
		model_lock.zero_grads(&mut gradients);
		drop(model_lock);
		drop(optimizer_lock);
		report_done.send((avg_loss, correct)).unwrap();
	}
}

fn test() {
	let dev = get_cpu();
	let mut model = dev.build_module::<SimpleTransformer, f32>();
	let data = Data::new();

	model
		.load("checkpoints/simple-transformer-00337.npz")
		.unwrap();

	let model = model;

	let test_data = data.test_data;

	let test_data_size = test_data.len();

	println!("Test data size: {test_data_size}");

	let correct = test_data
		.iter()
		.par_bridge()
		.filter(|Entry { tokens, label }| {
			let pred = model.forward(tokens.clone()).array()[0];
			pred.is_sign_positive() && *label || pred.is_sign_negative() && !*label
		})
		.count();

	println!("{:.1}%", correct as f32 / test_data_size as f32 * 100.0);
}

fn inference() {
	let dev = get_cpu();

	let mut model = dev.build_module::<SimpleTransformer, f32>();
	let data = Arc::new(Data::new());


	model
		.load("checkpoints/simple-transformer-00362.npz")
		.unwrap();

	let stdin = std::io::stdin();
	let mut stdout = std::io::stdout();
	let mut input = String::new();
	loop {
		input.clear();

		print!("Review> ");
		stdout.flush().unwrap();
		stdin.read_line(&mut input).unwrap();
		input = input.trim().to_string();
		let tokens = data.tokenize(&input);
		println!(
			"Tokens: {} ({})",
			data.tokenizer
				.tokenize(&input)
				.into_iter()
				.map(|t| t.to_string())
				.join(", "),
			tokens.iter().map(|t| t.to_string()).join(", ")
		);
		let shape = (tokens.len(),);
		let prediction = model
			.forward(dev.tensor_from_vec(tokens, shape))
			.sigmoid()
			.array()[0];
		println!("{:.1}%", prediction * 100.0);
		println!();
	}
}

fn train() {
	let mut model = get_cpu().build_module::<SimpleTransformer, f32>();
	model.reset_params();
	let data = Arc::new(Data::new());

	println!("Trainable parameter: {}", model.num_trainable_params());

	let mut best_training_loss = f32::INFINITY;
	let mut best_training_loss_epoch = 0;

	let optimizer = Arc::new(RwLock::new(Adam::<_, f32, Cpu>::new(&model, AdamConfig {
		lr: 1e-5,
		..Default::default()
	})));
	let model = Arc::new(RwLock::new(model));

	println!("Training data size: {}", data.training_data.len());
	println!();
	println!();

	let training_data = Arc::new(
		data.training_data
			.shuffled(&mut thread_rng())
			.cloned()
			.collect_vec(),
	);
	let data_left = Arc::new(AtomicUsize::new(0));

	let (barrier_forward, report_done_receiver) = {
		let model = model.clone();
		let optimizer = optimizer.clone();
		let data_left = data_left.clone();
		let training_data = training_data.clone();
		let barrier_forward = Arc::new(Barrier::new(get_parallelism() + 1));
		let barrier_forward_cloned = barrier_forward.clone();
		let barrier_new_batch = Arc::new(Barrier::new(get_parallelism()));
		let data_index = Arc::new(AtomicUsize::new(0));

		let (report_done_sender, report_done_receiver) = channel();
		std::iter::repeat_with(move || {
			let training_data = training_data.clone();
			let model = model.clone();
			let report_done_sender = report_done_sender.clone();
			let data_left = data_left.clone();
			let data_index = data_index.clone();
			let barrier_forward = barrier_forward_cloned.clone();
			let barrier_new_batch = barrier_new_batch.clone();
			let optimizer = Arc::as_ptr(&optimizer) as usize;
			let model = model.clone();
			std::thread::spawn(move || {
				thread_fn(
					model,
					unsafe {
						&*(optimizer as *const RwLock<Adam<SimpleTransformerBuilt, f32, Cpu>>)
					},
					training_data,
					data_index,
					data_left,
					report_done_sender,
					barrier_forward,
					barrier_new_batch,
				)
			});
		})
		.take(get_parallelism())
		.collect_vec();
		(barrier_forward, report_done_receiver)
	};

	let mut stdout = std::io::stdout();
	let mut rng = thread_rng();

	for epoch in 0..=EPOCHS {
		let start = Instant::now();
		let mut loss = 0.0;
		let mut accuracy = 0.0;

		for i in 1..=DATA_PER_BATCH {
			data_left.store(BATCH_SIZE, Ordering::Relaxed);

			barrier_forward.wait();

			let (losses, accuracies) = report_done_receiver
				.iter()
				.take(get_parallelism())
				.reduce(|(l, a), (l2, a2)| (l + l2, a + a2))
				.unwrap();
			loss += losses;
			accuracy += accuracies as f32 / BATCH_SIZE as f32;

			let elapsed = start.elapsed();
			writeln!(
				stdout,
				"\x1b[F\x1b[KEpoch {epoch:EPOCH_SIZE$}/{EPOCHS}: \
				 {:DATA_PER_BATCH_SIZE$}/{DATA_PER_BATCH} |{}>{}| avg loss: {:.3}, accuracy: \
				 {:.1}%, elapsed: {:.1?}, ETA: {:.1?}",
				i,
				"=".repeat(1 + 10 * i / DATA_PER_BATCH),
				" ".repeat(10 - 10 * i / DATA_PER_BATCH),
				loss / i as f32,
				accuracy / i as f32 * 100.0,
				elapsed,
				Duration::from_secs_f32(
					elapsed.as_secs_f32() * (DATA_PER_BATCH as f32 / i as f32 - 1.0)
				),
			)
			.unwrap();
			stdout.flush().unwrap();

			// threads will auto-sync again
		}
		let model = model.write().unwrap();
		let correct = data
			.validation_data
			.shuffled(&mut rng)
			.take(VALIDATION_BATCH_SIZE)
			.par_bridge()
			.map(|Entry { tokens, label }| {
				let pred = model.forward(tokens.clone()).array()[0];
				pred.is_sign_positive() && *label || pred.is_sign_negative() && !*label
			})
			.filter(|c| *c)
			.count();

		writeln!(
			stdout,
			"\x1b[F\x1b[KEpoch {epoch:EPOCH_SIZE$}/{EPOCHS}: \
			 {:DATA_PER_BATCH_SIZE$}/{DATA_PER_BATCH} |{}>| avg loss: {:.3}, accuracy: {:.1}%, \
			 validation_accuracy: {:.1}%, elapsed: {:.1?}",
			DATA_PER_BATCH,
			"=".repeat(1 + 10),
			loss / DATA_PER_BATCH as f32,
			accuracy / DATA_PER_BATCH as f32 * 100.0,
			correct as f32 / VALIDATION_BATCH_SIZE as f32 * 100.0,
			start.elapsed(),
		)
		.unwrap();
		model
			.save(format!("checkpoints/simple-transformer-{epoch:05}.npz"))
			.unwrap();

		if loss < best_training_loss {
			best_training_loss = loss;
			best_training_loss_epoch = epoch;
		}

		if epoch - best_training_loss_epoch > NO_LOSS_IMPROVEMENT {
			best_training_loss_epoch = epoch;
			optimizer.write().unwrap().cfg.lr *= 0.1;

			writeln!(stdout, "Reducing LR because reached a plateau.").unwrap();
		}

		writeln!(stdout).unwrap();
	}
}

fn main() {
	// let device = get_cpu();
	// device
	// 	.build_module::<lsh::builder::LshAttention<10, 0, 2>, f32>()
	// 	.forward(device.tensor([
	// 		[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0],
	// 		[2.0, 3.0, 5.0, 6.0, 5.0, 7.0, 9.0, 8.0, 0.0, -1.1],
	// 		[-1.0, -2.0, 5.0, -5.0, 4.0, 8.0, -9.0, 6.0, 4.0, 6.0],
	// 	]));
	if cfg!(feature = "train") {
		train();
	} else if cfg!(feature = "inference") {
		inference();
	} else if cfg!(feature = "test") {
		test();
	} else {
		panic!("Choose either `train` or `inference`.");
	}
}
