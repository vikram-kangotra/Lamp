use std::{fmt::Display, ops::{Add, Deref, DerefMut, Div, Mul, Neg, Sub}};

use crate::tensor::Tensor;

use rand::Rng;

pub struct Parameter {
    tensor: Tensor<f32>,
}

impl Parameter {
    pub fn new(shape: &[usize]) -> Self {

        let mut rng = rand::thread_rng();

        let data: Vec<f32> = (0..shape.iter().product())
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();

        Self {
            tensor: Tensor::new(&data, shape, true)
        }
    }
}

impl Deref for Parameter {
    type Target = Tensor<f32>;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

impl DerefMut for Parameter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tensor
    }
}

impl Into<Parameter> for Tensor<f32> {
    fn into(self) -> Parameter {
        Parameter {
            tensor: self
        }
    }
}

impl Display for Parameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Parameter: {}", self.tensor)
    }
}

impl Neg for &Parameter {
    type Output = Parameter;

    fn neg(self) -> Self::Output {
        (-&self.tensor).into()
    }
}

impl Add<&Parameter> for &Parameter {
    type Output = Parameter;

    fn add(self, other: &Parameter) -> Self::Output {
        (&self.tensor + &other.tensor).into()
    }
}

impl Add<f32> for &Parameter {
    type Output = Parameter;

    fn add(self, other: f32) -> Self::Output {
        (&self.tensor + other).into()
    }
}

impl Sub<&Parameter> for &Parameter {
    type Output = Parameter;

    fn sub(self, other: &Parameter) -> Self::Output {
        (&self.tensor - &other.tensor).into()
    }
}

impl Sub<f32> for &Parameter {
    type Output = Parameter;

    fn sub(self, other: f32) -> Self::Output {
        (&self.tensor - other).into()
    }
}

impl Mul<&Parameter> for &Parameter {
    type Output = Parameter;

    fn mul(self, other: &Parameter) -> Self::Output {
        (&self.tensor * &other.tensor).into()
    }
}

impl Mul<f32> for &Parameter {
    type Output = Parameter;

    fn mul(self, other: f32) -> Self::Output {
        (&self.tensor * other).into()
    }
}

impl Div<&Parameter> for &Parameter {
    type Output = Parameter;

    fn div(self, other: &Parameter) -> Self::Output {
        (&self.tensor / &other.tensor).into()
    }
}

impl Div<f32> for &Parameter {
    type Output = Parameter;

    fn div(self, other: f32) -> Self::Output {
        (&self.tensor / other).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parameter() {
        let p = Parameter::new(&[2, 2]);

        assert_eq!(p.get_shape(), &[2, 2]);
    }
}


