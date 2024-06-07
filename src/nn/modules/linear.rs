use std::{cell::RefCell, rc::Rc};

use crate::nn::parameter::Parameter;

use super::{Module, ModuleParams};

pub struct Linear {
    weight: Rc<RefCell<Parameter>>,
    bias: Rc<RefCell<Parameter>>,
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let weight = Rc::new(RefCell::new(Parameter::new(&[output_dim, input_dim])));
        let bias = Rc::new(RefCell::new(Parameter::new(&[output_dim, 1])));
        Self {
            weight,
            bias,
        }
    }
}

impl Module for Linear {

    fn forward(&self, input: &Parameter) -> Parameter {
        let weight: Parameter = self.weight.borrow().to_owned().into();
        let bias: Parameter = self.bias.borrow().to_owned().into();

        let output: Parameter = weight.mm(input).into();
        let output = &output + &bias;
        output
    }
}

impl ModuleParams for Linear {
    fn parameters(&self) -> Vec<Rc<RefCell<Parameter>>> {
        vec![
            self.weight.clone(),
            self.bias.clone(),
        ]
    }
}
