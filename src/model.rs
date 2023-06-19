use dfdx::prelude::*;

use crate::dataset::{AMOUNT_STARS, MAX_LEN};
use crate::get_cpu;
use crate::lsh::builder::LshAttention;

pub const VOCAB: usize = 8192;
pub const DIM: usize = 64;
pub const NUM_MHA_HEADS: usize = 4;
pub const NUM_ENCODER_LAYERS: usize = 4;
pub const HASHES: usize = 5;

pub struct PositionalEncoding {
	cached: Tensor<Rank2<MAX_LEN, DIM>, f32, Cpu>,
}
impl PositionalEncoding {
	fn new(dev: &Cpu) -> Self {
		Self{cached:dev.tensor_from_vec((0..MAX_LEN).flat_map(|pos| (0..DIM).map(move |i| if i % 2 == 0 {std::f32::consts::FRAC_PI_2}else {0.0} + if i % 2 == 0 {-1.0}else {1.0} * pos as f32 / 10000.0_f32.powf(2.0 * i as f32 / DIM as f32))).collect::<Vec<_>>(), (Const::<MAX_LEN>, Const::<DIM>))}
	}
}
impl<T: Tape<f32, Cpu>> Module<Tensor<(usize, Const<DIM>), f32, Cpu, T>> for PositionalEncoding {
	type Error = <Cpu as HasErr>::Err;
	type Output = Tensor<(usize, Const<DIM>), f32, Cpu, T>;

	fn try_forward(
		&self,
		input: Tensor<(usize, Const<DIM>), f32, Cpu, T>,
	) -> Result<Self::Output, Self::Error> {
		let encodings = if input.shape().0 <= MAX_LEN {
			self.cached.clone().slice((..input.shape().0, ..))
		} else {
			input.device().tensor_from_vec((0..input.shape().0).flat_map(|pos| (0..DIM).map(move |i| if i % 2 == 0 {std::f32::consts::FRAC_PI_2}else {0.0} + if i % 2 == 0 {-1.0}else {1.0} * pos as f32 / 10000.0_f32.powf(2.0 * i as f32 / DIM as f32))).collect::<Vec<_>>(), (input.shape().0, Const::<DIM>))
		};
		Ok(input + encodings)
	}
}
impl TensorCollection<f32, Cpu> for PositionalEncoding {
	type To<E2: Dtype, D2: Device<E2>> = Self;

	fn iter_tensors<V: ModuleVisitor<Self, f32, Cpu>>(
		visitor: &mut V,
	) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
		visitor.visit_fields((), |()| Self::new(&get_cpu()))
	}
}
impl BuildOnDevice<Cpu, f32> for PositionalEncoding {
	type Built = Self;

	fn try_build_on_device(device: &Cpu) -> Result<Self::Built, <Cpu as HasErr>::Err> {
		Ok(Self::new(device))
	}
}
pub struct TakeFirst;
impl<T: Tape<f32, Cpu>> Module<Tensor<(usize, Const<DIM>), f32, Cpu, T>> for TakeFirst {
	type Error = <Cpu as HasErr>::Err;
	type Output = Tensor1D<DIM, T>;

	fn try_forward(
		&self,
		input: Tensor<(usize, Const<DIM>), f32, Cpu, T>,
	) -> Result<Self::Output, Self::Error> {
		let index = input.device().try_tensor(0_usize)?;
		input.try_select(index)?.try_reshape()
	}
}
impl TensorCollection<f32, Cpu> for TakeFirst {
	type To<E2: Dtype, D2: Device<E2>> = Self;

	fn iter_tensors<V: ModuleVisitor<Self, f32, Cpu>>(
		visitor: &mut V,
	) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
		visitor.visit_fields((), |()| Self)
	}
}
impl BuildOnDevice<Cpu, f32> for TakeFirst {
	type Built = Self;

	fn try_build_on_device(_device: &Cpu) -> Result<Self::Built, <Cpu as HasErr>::Err> {
		Ok(Self)
	}
}
pub type SimpleTransformer = (
	(Embedding<VOCAB, DIM>, PositionalEncoding),
	Repeated<
		(
			LshAttention<DIM, NUM_MHA_HEADS, HASHES>,
			LayerNorm1D<DIM>,
			Residual<(Linear<DIM, DIM>, ReLU, Linear<DIM, DIM>)>,
			LayerNorm1D<DIM>,
		),
		NUM_ENCODER_LAYERS,
	>,
	TakeFirst,
	Linear<DIM, AMOUNT_STARS>,
);
pub type SimpleTransformerBuilt = <SimpleTransformer as BuildOnDevice<Cpu, f32>>::Built;
