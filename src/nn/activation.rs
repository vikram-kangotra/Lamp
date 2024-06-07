use super::parameter::Parameter;

pub trait Activation {
    fn forward(&self, x: &Parameter) -> Parameter;
}

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
