use dfdx::prelude::*;

use crate::dataset::{AMOUNT_STARS, MAX_LEN};
use crate::lsh::builder::LshAttention;

pub const VOCAB: usize = 8192;
pub const DIM: usize = 512;
pub const NUM_MHA_HEADS: usize = 8;
pub const NUM_ENCODER_LAYERS: usize = 8;
pub const HASHES: usize = 8;

pub mod builder {
    pub struct PositionalEncoding;
}

pub struct PositionalEncoding<D: Device<f32>> {
    cached: Tensor<Rank2<MAX_LEN, DIM>, f32, D>,
}
impl<D: Device<f32>> PositionalEncoding<D> {
    fn new(dev: &D) -> Self {
        Self{cached:dev.tensor_from_vec((0..MAX_LEN).flat_map(|pos| (0..DIM).map(move |i| if i % 2 == 0 {std::f32::consts::FRAC_PI_2}else {0.0} + if i % 2 == 0 {-1.0}else {1.0} * pos as f32 / 10000.0_f32.powf(2.0 * i as f32 / DIM as f32))).collect::<Vec<_>>(), (Const::<MAX_LEN>, Const::<DIM>))}
    }
}

impl<D: Device<f32>> BuildOnDevice<D, f32> for builder::PositionalEncoding {
    type Built = PositionalEncoding<D>;

    fn try_build_on_device(device: &D) -> Result<Self::Built, <D as HasErr>::Err> {
        Ok(Self::Built::new(device))
    }
}
impl<D: Device<f32>, T: Tape<f32, D>> Module<Tensor<(usize, Const<DIM>), f32, D, T>>
    for PositionalEncoding<D>
{
    type Error = <D as HasErr>::Err;
    type Output = Tensor<(usize, Const<DIM>), f32, D, T>;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<DIM>), f32, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let encodings = if input.shape().0 <= MAX_LEN {
            self.cached.clone().slice((..input.shape().0, ..))
        } else {
            input.device().tensor_from_vec((0..input.shape().0).flat_map(|pos| (0..DIM).map(move |i| if i % 2 == 0 {std::f32::consts::FRAC_PI_2}else {0.0} + if i % 2 == 0 {-1.0}else {1.0} * pos as f32 / 10000.0_f32.powf(2.0 * i as f32 / DIM as f32))).collect::<Vec<_>>(), (input.shape().0, Const::<DIM>))
        };
        Ok(input + encodings)
    }
}
impl<D: Default + Device<f32>> TensorCollection<f32, D> for PositionalEncoding<D> {
    type To<E2: Dtype, D2: Device<E2>> = Self;

    fn iter_tensors<V: ModuleVisitor<Self, f32, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields((), |()| Self::new(&D::default()))
    }
}
pub struct TakeFirst;
impl<D: Device<f32>, T: Tape<f32, D>> Module<Tensor<(usize, Const<DIM>), f32, D, T>> for TakeFirst {
    type Error = <D as HasErr>::Err;
    type Output = Tensor<Rank1<DIM>, f32, D, T>;

    fn try_forward(
        &self,
        input: Tensor<(usize, Const<DIM>), f32, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let index = input.device().try_tensor(0_usize)?;
        input.try_select(index)?.try_reshape()
    }
}
impl<D: Device<f32>> TensorCollection<f32, D> for TakeFirst {
    type To<E2: Dtype, D2: Device<E2>> = Self;

    fn iter_tensors<V: ModuleVisitor<Self, f32, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields((), |()| Self)
    }
}
impl<D: Device<f32>> BuildOnDevice<D, f32> for TakeFirst {
    type Built = Self;

    fn try_build_on_device(_device: &D) -> Result<Self::Built, <D as HasErr>::Err> {
        Ok(Self)
    }
}
pub type SimpleTransformer = (
    (Embedding<VOCAB, DIM>, builder::PositionalEncoding),
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
