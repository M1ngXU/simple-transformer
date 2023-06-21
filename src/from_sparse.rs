use cudarc::driver::CudaSlice;
use dfdx::prelude::*;
use dfdx::tensor::cpu::index_to_i;
use itertools::Itertools;

pub trait FromSparse<
	E: Dtype,
	T: Tape<E, Self>,
	OutputShape: Shape<Concrete = [usize; OutputShape::NUM_DIMS]>,
>: Device<E>
{
	fn from_sparse(
		&self,
		values: Tensor<(usize,), E, Self, T>,
		indeces: Tensor<(usize, Const<{ OutputShape::NUM_DIMS }>), usize, Self>,
		output_shape: OutputShape,
	) -> Tensor<OutputShape, E, Self, T>;
}

impl<T: Tape<f32, Self>, OutputShape: Shape<Concrete = [usize; OutputShape::NUM_DIMS]>>
	FromSparse<f32, T, OutputShape> for Cpu
{
	fn from_sparse(
		&self,
		values: Tensor<(usize,), f32, Self, T>,
		indeces: Tensor<(usize, Const<{ OutputShape::NUM_DIMS }>), usize, Self>,
		output_shape: OutputShape,
	) -> Tensor<OutputShape, f32, Self, T> {
		assert_eq!(values.shape().0, indeces.shape().0);
		let (values, mut tape) = values.split_tape();
		let indeces: Vec<[usize; OutputShape::NUM_DIMS]> = indeces
			.data()
			.unwrap()
			.array_chunks::<{ OutputShape::NUM_DIMS }>()
			.copied()
			.collect_vec();
		let mut output = self.zeros_like(&output_shape);
		for (i, index) in indeces.iter().enumerate() {
			output[*index] = values[[i]];
		}
		let inp_ghost = values.clone();
		let out_ghost = output.clone();
		tape.add_backward_op(move |grads| {
			grads.try_alloc_for(&inp_ghost)?;
			grads.try_alloc_for(&out_ghost)?;
			let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);
			for (i, index) in indeces.into_iter().enumerate() {
				grad_inp[index_to_i(inp_ghost.shape(), &inp_ghost.strides(), [i])] +=
					grad_out[index_to_i(out_ghost.shape(), &out_ghost.strides(), index)];
			}
			Ok(())
		});
		output.put_tape(tape)
	}
}

#[cfg(feature = "cuda")]
impl<
		T: Tape<f32, Self> + 'static,
		OutputShape: Shape<Concrete = [usize; OutputShape::NUM_DIMS]>,
	> FromSparse<f32, T, OutputShape> for Cuda
{
	fn from_sparse(
		&self,
		values: Tensor<(usize,), f32, Self, T>,
		indeces: Tensor<(usize, Const<{ OutputShape::NUM_DIMS }>), usize, Self>,
		output_shape: OutputShape,
	) -> Tensor<OutputShape, f32, Self, T> {
		use cudarc::driver::LaunchAsync;

		assert_eq!(values.shape().0, indeces.shape().0);

		if !self.dev.has_func("from_sparse_f32", "from_sparse_fwd_f32") {
			self.dev
				.load_ptx("from_sparse_f32".into(), "from_sparse_f32", &[
					"from_sparse_fwd_f32",
					"from_sparse_bwd_f32",
				])
				.unwrap();
		}

		let fwd_fn = self
			.dev
			.get_func("from_sparse_f32", "from_sparse_fwd_f32")
			.unwrap();
		let bwd_fn = self
			.dev
			.get_func("from_sparse_f32", "from_sparse_bwd_f32")
			.unwrap();

		let (values, mut tape) = values.split_tape();

		let cfg = launch_cfg::<128>(values.shape().0 as u32);

		let mut values_info = Vec::with_capacity(1 * 2);
		values_info.push(values.shape().concrete()[0]);
		values_info.push(values.strides()[0]);
		let values_info = self.dev.htod_copy(values_info).unwrap();

		let mut indeces_info = Vec::with_capacity(2 * 2);
		indeces_info.extend(indeces.shape().concrete());
		indeces_info.extend(indeces.strides());
		let indeces_info = self.dev.htod_copy(indeces_info).unwrap();

		let mut output = self.dev.alloc_zeros(output_shape.num_elements()).unwrap();
		let mut output_info = Vec::with_capacity(OutputShape::NUM_DIMS * 2);
		output_info.extend(output_shape.concrete());
		output_info.extend(output_shape.strides());
		let output_info = self.dev.htod_copy(output_info).unwrap();

		let params: (
			usize,
			&CudaSlice<f32>,
			&CudaSlice<usize>,
			&CudaSlice<usize>,
			&CudaSlice<usize>,
			&mut CudaSlice<f32>,
			&CudaSlice<usize>,
			usize,
		) = (
			values.shape().0,
			&**values.data,
			&values_info,
			&**indeces.data,
			&indeces_info,
			&mut output,
			&output_info,
			OutputShape::NUM_DIMS,
		);
		unsafe {
			fwd_fn.launch(cfg, params).unwrap();
		}

		let inp_ghost = values.clone();
		let output = self.build_tensor(output_shape, output_shape.strides(), output);
		let out_ghost = output.clone();
		tape.add_backward_op(move |grads| {
			grads.try_alloc_for(&inp_ghost)?;
			grads.try_alloc_for(&out_ghost)?;
			let (grad_inp, grad_out) = grads.mut_and_ref(&inp_ghost, &out_ghost);

			let params: (
				usize,
				&mut CudaSlice<f32>,
				&CudaSlice<usize>,
				&CudaSlice<usize>,
				&CudaSlice<usize>,
				&CudaSlice<f32>,
				&CudaSlice<usize>,
				usize,
			) = (
				values.shape().0,
				&mut **grad_inp,
				&values_info,
				&**indeces.data().unwrap(),
				&indeces_info,
				&**grad_out,
				&output_info,
				OutputShape::NUM_DIMS,
			);
			unsafe {
				bwd_fn.launch(cfg, params).unwrap();
			}
			Ok(())
		});
		output.put_tape(tape)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn from_sparse() {
		let dev = AutoDevice::default();

		let indeces = dev
			.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]])
			.reshape_like(&(5, Const::<2>));
		let values = dev.tensor([0.0, 1.0, 2.0, 3.0, 4.0]).reshape_like(&(5,));

		let (output, mut tape) = dev
			.from_sparse(values.clone().leaky_traced(), indeces, (5, 5))
			.split_tape();

		let expected = vec![
			0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0,
			0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0,
		];

		assert_eq!(output.as_vec(), expected);

		dev.dev
			.htod_copy_into(
				(0..25).map(|g| g as f32).collect_vec(),
				&mut **tape.gradients.get_or_alloc_mut(&output).unwrap(),
			)
			.unwrap();

		let grads = tape.execute().unwrap();

		let gradients = grads.get(&values);
		let expected = vec![0.0, 6.0, 12.0, 18.0, 24.0];
		assert_eq!(gradients.as_vec(), expected);
	}
}
