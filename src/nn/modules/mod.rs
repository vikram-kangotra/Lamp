use std::{cell::RefCell, rc::Rc};

use super::parameter::Parameter;

pub mod linear;

pub trait Module : ModuleParams {
    fn forward(&self, input: &Parameter) -> Parameter;

    fn to_gpu(&mut self) {
        for parameter in &self.parameters() {
            parameter.borrow_mut().to_gpu().unwrap();
        }
    }

    fn to_cpu(&mut self) {
        for parameter in &self.parameters() {
            parameter.borrow_mut().to_cpu().unwrap();
        }
    }
}

pub trait ModuleParams {
    fn parameters(&self) -> Vec<Rc<RefCell<Parameter>>>;
}
