use std::{cell::RefCell, rc::Rc};

use crate::tensor::{FromRcRefCell, Tensor, TensorData, TensorElement};

use super::AutogradFunction;

#[derive(Debug)]
pub struct ReluBackward<T: TensorElement> {
    inputs: [Tensor<T>; 1],
}

impl<T: TensorElement> ReluBackward<T> {
    pub fn new(inputs: [Rc<RefCell<TensorData<T>>>; 1]) -> Self {
        let inputs = [
            Tensor::<T>::from_rc_refcell(&inputs[0]),
        ];
        Self { inputs }
    }
}

impl<T: TensorElement> AutogradFunction<T> for ReluBackward<T> {
    fn inputs(&self) -> &[Tensor<T>] {
        &self.inputs
    }

    fn backward(&self, grad: Tensor<T>) -> Vec<Tensor<T>> {
        let x = &self.inputs[0];
        let mask = x.gt(&Tensor::zeros_like(x));
        vec![&grad * &mask]
    }
}
