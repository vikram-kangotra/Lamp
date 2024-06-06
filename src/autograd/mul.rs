use std::{cell::RefCell, rc::Rc};

use crate::tensor::{FromRcRefCell, Tensor, TensorData, TensorElement};

use super::AutogradFunction;

#[derive(Debug)]
pub struct MulBackward<T: TensorElement> {
    inputs: [Tensor<T>; 2],
}

impl<T: TensorElement> MulBackward<T> {
    pub fn new(inputs: [Rc<RefCell<TensorData<T>>>; 2]) -> Self {
        let inputs = [
            Tensor::<T>::from_rc_refcell(&inputs[0]),
            Tensor::<T>::from_rc_refcell(&inputs[1]),
        ];
        Self { inputs }
    }
}

impl<T: TensorElement> AutogradFunction<T> for MulBackward<T> {
    fn inputs(&self) -> &[Tensor<T>] {
        &self.inputs
    }

    fn backward(&self, grad: Tensor<T>) -> Vec<Tensor<T>> {

        let x = &self.inputs[0];
        let y = &self.inputs[1];

        let x_grad = &grad * y;
        let y_grad = &grad * x;

        vec![x_grad, y_grad]
    }
}
