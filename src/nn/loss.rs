use super::parameter::Parameter;

pub struct MSELoss;

impl MSELoss {

    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: &Parameter, target: &Parameter) -> Parameter {
        let diff = input - target;
        let loss = &diff * &diff;
        loss.sum().into()
    }
}
