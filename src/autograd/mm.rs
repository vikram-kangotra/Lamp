use std::{cell::RefCell, rc::Rc};

use crate::tensor::{FromRcRefCell, Tensor, TensorData, TensorElement};

use super::AutogradFunction;

#[derive(Debug)]
pub struct MMBackward<T: TensorElement> {
    inputs: [Tensor<T>; 2],
}

impl<T: TensorElement> MMBackward<T> {
    pub fn new(inputs: [Rc<RefCell<TensorData<T>>>; 2]) -> Self {
        let inputs = [
            Tensor::<T>::from_rc_refcell(&inputs[0]),
            Tensor::<T>::from_rc_refcell(&inputs[1]),
        ];
        Self { inputs }
    }
}

impl<T: TensorElement> AutogradFunction<T> for MMBackward<T> {
    fn inputs(&self) -> &[Tensor<T>] {
        &self.inputs
    }

    fn backward(&self, grad: Tensor<T>) -> Vec<Tensor<T>> {
        let x = &self.inputs[0];
        let y = &self.inputs[1];

        let x_grad = y.transpose().mm(&grad);
        let y_grad = grad.mm(&x.transpose());

        vec![x_grad, y_grad]
    }
}
