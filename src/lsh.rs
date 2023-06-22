use dfdx::nn::modules::Linear;
use dfdx::prelude::*;
use itertools::Itertools;
use num_traits::Float;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Distribution, StandardNormal};

pub mod builder {
	#[derive(Debug, Clone)]
	pub struct LshAttention<
		const EMBED_DIM: usize,
		const NUM_HEADS: usize,
		const HASHES: usize,
		const K_DIM: usize = EMBED_DIM,
		const V_DIM: usize = EMBED_DIM,
	>;
	impl<const M: usize, const H: usize, const K: usize, const V: usize, const HASHES: usize>
		LshAttention<M, H, HASHES, K, V>
	{
		pub const TYPE_CHECK: () = assert!(
			K % H == 0 && V % H == 0,
			"NUM_HEADS must divide K_DIM & V_DIM evenly! If you haven't specified K_DIM & V_DIM, \
			 they default to EMBED_DIM, which means NUM_HEADS must divide EMBED_DIM evenly."
		);
	}
}

impl<
		const M: usize,
		const H: usize,
		const K: usize,
		const V: usize,
		const HASHES: usize,
		E: Dtype,
		D: Device<E>,
	> BuildOnDevice<D, E> for builder::LshAttention<M, H, HASHES, K, V>
where
	LshAttention<M, H, HASHES, K, V, E, D>: BuildModule<D, E>,
{
	type Built = LshAttention<M, H, HASHES, K, V, E, D>;

	fn build_on_device(device: &D) -> Self::Built {
		#[allow(clippy::let_unit_value)]
		let _ = Self::TYPE_CHECK;
		Self::Built::build(device)
	}
}

#[derive(Debug, Clone)]
pub struct LshAttention<
	const EMBED_DIM: usize,
	const NUM_HEADS: usize,
	const HASHES: usize,
	const K_DIM: usize,
	const V_DIM: usize,
	E: Dtype,
	D: Storage<E>,
> {
	pub w_qk: Linear<EMBED_DIM, K_DIM, E, D>,
	pub w_v:  Linear<EMBED_DIM, V_DIM, E, D>,
	pub w_o:  Linear<V_DIM, EMBED_DIM, E, D>,
}

impl<
		const M: usize,
		const H: usize,
		const K: usize,
		const V: usize,
		const HASHES: usize,
		E: Dtype + Float + SampleUniform,
		D: Device<E>,
	> TensorCollection<E, D> for LshAttention<M, H, HASHES, K, V, E, D>
{
	type To<E2: Dtype, D2: Device<E2>> = LshAttention<M, H, HASHES, K, V, E2, D2>;

	fn iter_tensors<Vi: ModuleVisitor<Self, E, D>>(
		visitor: &mut Vi,
	) -> Result<Option<Self::To<Vi::E2, Vi::D2>>, Vi::Err> {
		visitor.visit_fields(
			(
				Self::module("w_qk", |s| &s.w_qk, |s| &mut s.w_qk),
				Self::module("w_v", |s| &s.w_v, |s| &mut s.w_v),
				Self::module("w_o", |s| &s.w_o, |s| &mut s.w_o),
			),
			|(w_qk, w_v, w_o)| LshAttention { w_qk, w_v, w_o },
		)
	}
}

impl<
		const M: usize,
		const H: usize,
		const K: usize,
		const V: usize,
		const HASHES: usize,
		D: Device<E> + FromSparse<E, T, (usize, usize)> + ZerosTensor<usize>,
		E: Dtype,
		T: Tape<E, D>,
	>
	Module<(
		Tensor<(usize, Const<M>), E, D, T>,
		Tensor<(usize, Const<M>), E, D>,
	)> for LshAttention<M, H, HASHES, K, V, E, D>
where
	StandardNormal: Distribution<E>,
{
	type Error = D::Err;
	type Output = Tensor<(usize, Const<M>), E, D, T>;

	fn try_forward(
		&self,
		input: (
			Tensor<(usize, Const<M>), E, D, T>,
			Tensor<(usize, Const<M>), E, D>,
		),
	) -> Result<Self::Output, Self::Error> {
		Ok(self.forward(input))
	}

	fn forward(
		&self,
		(qk, v): (
			Tensor<(usize, Const<M>), E, D, T>,
			Tensor<(usize, Const<M>), E, D>,
		),
	) -> Self::Output {
		assert_eq!(qk.shape().num_elements() / M, v.shape().num_elements() / M);

		let dev = qk.device().clone();

		let s1 = qk.shape().0;

		let v = self.w_v.forward(v.retaped::<T>());

		let (q, tape) = self.w_qk.forward(qk).split_tape();

		let planes = dev.sample_normal_like(&(Const::<K>, Const::<HASHES>));

		let buckets = q.clone().matmul(planes).ge(E::from_f64(0.0).unwrap());

		let bucket_ids = buckets
			.as_vec()
			.chunks(HASHES)
			.map(|hash| hash.iter().fold(0_usize, |a, &b| (a << 1) | b as usize))
			.enumerate()
			.map(|(i, hash)| (hash, i))
			.into_group_map();

		let mut indeces = Vec::new();
		let mut values = dev.zeros_like(&(0,)).put_tape(tape);

		for (_, idc) in bucket_ids {
			let shape = (idc.len(),);
			let idc_iter = idc.iter().copied();
			indeces.extend(
				idc_iter
					.clone()
					.cartesian_product(idc_iter)
					.flat_map(|(a, b)| [a, b]),
			);
			let idc = dev.tensor_from_vec(idc, shape);
			let (q, tape) = q.clone().retaped::<T>().gather(idc.clone()).split_tape();
			let (k, tape) = q
				.clone()
				.put_tape(tape)
				.normalize::<Axis<1>>(1e-6)
				.permute::<_, Axes2<1, 0>>()
				.split_tape();
			let sparse_attention =
				q.put_tape(tape).matmul(k) / E::from_f64((K as f64).sqrt()).unwrap();
			let sparse_attention = sparse_attention.reshape_like(&(shape.0 * shape.0,));
			values = (values, sparse_attention).concat_along(Axis::<0>);
		}
		let indeces_shape = (indeces.len() / 2, Const::<2>);
		let weights = dev.from_sparse(
			values,
			dev.tensor_from_vec(indeces, indeces_shape),
			(s1, s1),
		);
		let weights = weights.softmax::<Axis<1>>();

		let tokens = weights.matmul(v);

		self.w_o.forward(tokens)
	}
}

impl<
		const M: usize,
		const H: usize,
		const K: usize,
		const V: usize,
		const HASHES: usize,
		D: Device<f32>,
		Src,
	> Module<Src> for LshAttention<M, H, HASHES, K, V, f32, D>
where
	f32: Dtype,
	Src: SplitTape,
	StandardNormal: Distribution<f32>,
	Self: Module<(Src, Src::NoTape), Output = Src, Error = <D as HasErr>::Err>,
{
	type Error = <D as HasErr>::Err;
	type Output = Src;

	fn try_forward(&self, src: Src) -> Result<Self::Output, Self::Error> {
		let (src, tape) = src.split_tape();
		self.try_forward((src.clone().put_tape(tape), src))
	}
}

impl<
		const M: usize,
		const H: usize,
		const K: usize,
		const V: usize,
		const HASHES: usize,
		D: Device<E>,
		E: Dtype,
	> NonMutableModule for LshAttention<M, H, HASHES, K, V, E, D>
{
}
