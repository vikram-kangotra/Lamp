use std::{cell::RefCell, rc::Rc};

use super::parameter::Parameter;

pub mod sgd;

pub trait Optimizer {
    fn step(&mut self);

    fn zero_grad(&mut self) {
        for parameter in &self.parameters() {
            parameter.borrow_mut().zero_grad();
        }
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Parameter>>>;
}
