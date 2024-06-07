use super::parameter::Parameter;
use serde::{Serialize, Deserialize};

pub trait Activation: Serialize {
    fn forward(&self, x: &Parameter) -> Parameter;
}

#[derive(Serialize, Deserialize)]
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for ReLU {
    fn forward(&self, x: &Parameter) -> Parameter {
        x.relu().into()
    }
}

#[derive(Serialize, Deserialize)]
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Self
    }
}

impl Activation for Sigmoid {
    fn forward(&self, x: &Parameter) -> Parameter {
        x.sigmoid().into()
    }
}
