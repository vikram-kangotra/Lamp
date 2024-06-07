use std::{cell::RefCell, fs::File, rc::Rc};

use super::parameter::Parameter;

use serde::{Serialize, Deserialize};
use std::io::{Write, Read};
use std::result::Result;
use std::error::Error;

pub mod linear;

pub trait Module : ModuleParams + Serialize + for<'de> Deserialize<'de> {
    fn forward(&self, input: &Parameter) -> Parameter;

    fn to_gpu(&mut self) {
        for parameter in &self.parameters() {
            parameter.borrow_mut().to_gpu().unwrap();
        }
    }

    fn to_cpu(&mut self) {
        for parameter in &self.parameters() {
            parameter.borrow_mut().to_cpu().unwrap();
        }
    }

    fn save_model(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let encoded = serde_json::to_string(self).unwrap();
        let mut file = File::create(path)?;
        file.write_all(encoded.as_bytes())?;
        Ok(())
    }

    fn load_model(path: &str) -> Result<Self, Box<dyn Error>>
    where
        Self: Sized,
    {
        let mut file = File::open(path)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;
        let model: Self = serde_json::from_slice(&buffer)?;
        Ok(model)
    }
}

pub trait ModuleParams {
    fn parameters(&self) -> Vec<Rc<RefCell<Parameter>>>;
}
