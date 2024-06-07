use std::{cell::RefCell, rc::Rc};

use lamp::nn::{activation::{Activation, ReLU, Sigmoid}, loss::MSELoss, modules::{linear::Linear, Module, ModuleParams}, optim::{sgd::SGD, Optimizer}, parameter::Parameter};
use lamp::tensor::Tensor;

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct Model {
    fc1: Linear,
    fc2: Linear,
    relu: ReLU,
    sigmoid: Sigmoid,
}

impl Model {
    fn new(input_dim: usize, output_dim: usize) -> Self {
        let fc1 = Linear::new(input_dim, 10);
        let fc2 = Linear::new(10, output_dim);
        let relu = ReLU::new();
        let sigmoid = Sigmoid::new();

        Self { fc1, fc2, relu, sigmoid }
    }
}

impl Module for Model {

    fn forward(&self, input: &Parameter) -> Parameter {
        let x = self.fc1.forward(input);
        let x = self.relu.forward(&x);
        let x = self.fc2.forward(&x);
        self.sigmoid.forward(&x)
    }
}

impl ModuleParams for Model {

    fn parameters(&self) -> Vec<Rc<RefCell<Parameter>>> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
}

fn main() {

    let epochs = 100;

    let model = Model::new(2, 1);
    let criterion = MSELoss::new();
    let mut optim = SGD::new(model.parameters(), 0.5, 0.0);

    let input = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let target = input.iter().map(|[a, b]| (*a as u32 ^ *b as u32) as f32).collect::<Vec<f32>>();

    let mut loss = Parameter::new(&[1, 1]);

    for epoch in 0..epochs {
        for (x, target) in input.iter().zip(target.iter()) {
            let x: Parameter = Tensor::<f32>::new(x, &[2, 1], false).into();
            let target: Parameter = Tensor::<f32>::new(&[*target], &[1, 1], false).into();

            let y = model.forward(&x);
            loss = criterion.forward(&y, &target);

            optim.zero_grad();
            loss.backward(None);
            optim.step();
        }

        println!("Epoch: {}/{epochs}, Loss: {}", epoch + 1, loss);
    }

    println!("Final loss: {}", loss.get_flat_item(0));

    model.save_model("examples/model").unwrap();

    let model = Model::load_model("examples/model").unwrap();

    let x = Tensor::<f32>::new(&[0.0, 0.0], &[2, 1], false).into();
    let y = model.forward(&x);
    println!("0 XOR 0 = {}", if y.get_flat_item(0) > 0.5 { 1 } else { 0 });

    let x = Tensor::<f32>::new(&[0.0, 1.0], &[2, 1], false).into();
    let y = model.forward(&x);
    println!("0 XOR 1 = {}", if y.get_flat_item(0) > 0.5 { 1 } else { 0 });

    let x = Tensor::<f32>::new(&[1.0, 0.0], &[2, 1], false).into();
    let y = model.forward(&x);
    println!("1 XOR 0 = {}", if y.get_flat_item(0) > 0.5 { 1 } else { 0 });

    let x = Tensor::<f32>::new(&[1.0, 1.0], &[2, 1], false).into();
    let y = model.forward(&x);
    println!("1 XOR 1 = {}", if y.get_flat_item(0) > 0.5 { 1 } else { 0 });

}
