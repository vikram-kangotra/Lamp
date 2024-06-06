use std::{cell::RefCell, rc::Rc};

use crate::tensor::{FromRcRefCell, Tensor, TensorData, TensorElement};

use super::AutogradFunction;

#[derive(Debug)]
pub struct SumBackward<T: TensorElement> {
    inputs: [Tensor<T>; 1],
}

impl<T: TensorElement> SumBackward<T> {
    pub fn new(inputs: [Rc<RefCell<TensorData<T>>>; 1]) -> Self {
        let inputs = [
            Tensor::<T>::from_rc_refcell(&inputs[0]),
        ];
        Self { inputs }
    }
}

impl<T: TensorElement> AutogradFunction<T> for SumBackward<T> {
    fn inputs(&self) -> &[Tensor<T>] {
        &self.inputs
    }

    fn backward(&self, grad: Tensor<T>) -> Vec<Tensor<T>> {
        vec![&self.inputs[0].ones_like() * grad.get_flat_item(0)]
    }
}
