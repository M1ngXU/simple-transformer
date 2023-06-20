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
