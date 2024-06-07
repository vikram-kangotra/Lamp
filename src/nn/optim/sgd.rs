use std::{cell::RefCell, rc::Rc};

use crate::nn::parameter::Parameter;

use super::Optimizer;

pub struct SGD {
    parameters: Vec<Rc<RefCell<Parameter>>>,
    lr: f64,
    momentum: f64,
    velocity: Vec<Parameter>,
}

impl SGD {
    pub fn new(parameters: Vec<Rc<RefCell<Parameter>>>, lr: f64, momentum: f64) -> Self {

        let velocity = parameters.iter().map(|p| p.borrow().zeros_like().into()).collect::<Vec<Parameter>>();

        Self {
            parameters,
            lr,
            momentum,
            velocity
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        let mut i = 0;
        for parameter in &self.parameters {
            let mut parameter = parameter.borrow_mut();
            let grad = parameter.grad().unwrap();
            let mut velocity = self.velocity[i].mul_scalar(self.momentum as f32);
            velocity = &velocity - &grad.mul_scalar(self.lr as f32);
            self.velocity[i] = velocity.into();
            *parameter += &self.velocity[i];

            parameter.detach();
            i += 1;
        }
    }

    fn parameters(&self) -> Vec<Rc<RefCell<Parameter>>> {
        self.parameters.clone()
    }
}
