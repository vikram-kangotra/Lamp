use std::{cell::RefCell, rc::Rc};

use crate::tensor::{FromRcRefCell, Tensor, TensorData, TensorElement};

use super::AutogradFunction;

#[derive(Debug)]
pub struct SubBackward<T: TensorElement> {
    inputs: [Tensor<T>; 2],
}

impl<T: TensorElement> SubBackward<T> {
    pub fn new(inputs: [Rc<RefCell<TensorData<T>>>; 2]) -> Self {
        let inputs = [
            Tensor::<T>::from_rc_refcell(&inputs[0]),
            Tensor::<T>::from_rc_refcell(&inputs[1]),
        ];
        Self { inputs }
    }
}

impl<T: TensorElement> AutogradFunction<T> for SubBackward<T> {
    fn inputs(&self) -> &[Tensor<T>] {
        &self.inputs
    }

    fn backward(&self, grad: Tensor<T>) -> Vec<Tensor<T>> {

        let x_grad = grad.clone();
        let y_grad = -&grad;

        vec![x_grad, y_grad]
    }
}
