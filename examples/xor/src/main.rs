use std::{cell::RefCell, rc::Rc};

use lamp::nn::{activation::{Activation, ReLU, Sigmoid}, loss::MSELoss, modules::{linear::Linear, Module, ModuleParams}, optim::{sgd::SGD, Optimizer}, parameter::Parameter};
use lamp::tensor::Tensor;

use serde::{Serialize, Deserialize};

use plotters::prelude::*;

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

fn loss_plot(loss: Vec<f32>) {
    let root = BitMapBackend::new("loss.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Loss", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..100, 0.0f32..1.0).unwrap();

    chart.configure_mesh().draw().unwrap();

    chart.draw_series(LineSeries::new(
        loss.iter().enumerate().map(|(i, y)| (i as i32, *y)),
        &RED,
    )).unwrap();

    root.present().unwrap();
}

fn color_map(model: &Model) {
    let root = BitMapBackend::new("color_map.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("Color Map", ("sans-serif", 30))
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0.0f32..1.0, 0.0f32..1.0).unwrap();

    chart.configure_mesh().draw().unwrap();

    for x in 0..200 {
        for y in 0..200 {
            let x = x as f32 / 200.0;
            let y = y as f32 / 200.0;

            let input = Tensor::<f32>::new(&[x, y], &[2, 1], false).into();
            let output = model.forward(&input);

            let color = if output.get_flat_item(0) > 0.5 { GREEN } else { BLUE };

            chart.draw_series(PointSeries::of_element(
                vec![(x, y)],
                5,
                &color,
                &|c, s, st| {
                    return EmptyElement::at(c)    // We want to use the provided coordinates
                        + Circle::new((0, 0), s, st.filled()); // And a circle that is 2 pixels in diameter
                },
            )).unwrap();
        }
    }

    root.present().unwrap();
}

fn main() -> Result<(), Box<dyn std::error::Error>> {

    let epochs = 1000;

    let model = Model::new(2, 1);
    let criterion = MSELoss::new();
    let mut optim = SGD::new(model.parameters(), 0.5, 0.0);

    let input = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let target = [0.0, 1.0, 1.0, 0.0];

    let mut loss = Parameter::new(&[1, 1]);
    let mut loss_series = Vec::new();

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

        loss_series.push(loss.get_flat_item(0));
        println!("Epoch: {}/{epochs}, Loss: {}", epoch + 1, loss);
    }

    println!("Final loss: {}", loss.get_flat_item(0));

    loss_plot(loss_series);
    color_map(&model);

    model.save_model("model").unwrap();

    let model = Model::load_model("model").unwrap();

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

    Ok(())
}
