pub mod neg;
pub mod add;
pub mod sub;
pub mod mul;
pub mod div;
pub mod sum;
pub mod transpose;
pub mod mm;
pub mod relu;
pub mod sigmoid;

use crate::autograd::{
    neg::NegBackward, add::AddBackward, sub::SubBackward,
    mul::MulBackward, div::DivBackward, sum::SumBackward, 
    transpose::TransposeBackward, mm::MMBackward, 
    relu::ReluBackward, sigmoid::SigmoidBackward
};

use crate::tensor::{Tensor, TensorElement};

macro_rules! autograd_ops {
    ($($op:ident),*) => {
        #[derive(Debug)]
        pub enum AutogradOps<T: TensorElement> {
            $(
                $op($op<T>),
            )*
        } 

        impl<T: TensorElement> AutogradOps<T> {
            pub fn inputs(&self) -> &[Tensor<T>] {
                match self {
                    $(
                        AutogradOps::$op(op) => op.inputs(),
                    )*
                }
            }

            pub fn backward(&self, grad: Tensor<T>) -> Vec<Tensor<T>> {
                match self {
                    $(
                        AutogradOps::$op(op) => op.backward(grad),
                        )*
                }
            }
        }

    };
}

autograd_ops!(
    NegBackward, AddBackward, SubBackward, 
    MulBackward, DivBackward, SumBackward, 
    TransposeBackward, MMBackward, 
    ReluBackward, SigmoidBackward
);

trait AutogradFunction<T: TensorElement> {
    fn inputs(&self) -> &[Tensor<T>];
    fn backward(&self, grad: Tensor<T>) -> Vec<Tensor<T>>;
}
