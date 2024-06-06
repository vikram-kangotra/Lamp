use super::parameter::Parameter;

pub trait Activation {
    fn forward(&self, x: &Parameter) -> Parameter;
}

pub struct ReLU;

impl Activation for ReLU {
    fn forward(&self, x: &Parameter) -> Parameter {
        x.relu().into()
    }
}
