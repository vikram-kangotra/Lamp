#![allow(dead_code)]
#![allow(unused_variables)]

mod autograd;
mod cuda_kernel;
mod nn;
mod tensor;

use nn::{loss::MSELoss, parameter::Parameter};
use tensor::Tensor;

fn main() {

    let criterion = MSELoss::new();

    let a: Parameter = Tensor::<f32>::new(&[1., 2., 3., 3., 2., 1.], &[2, 3], true).into();
    let b: Parameter = Tensor::<f32>::new(&[1., 2., 3., 3., 2., 1.], &[2, 3], true).into();

    let loss = criterion.forward(&a, &b);

    println!("{}", loss);
}
