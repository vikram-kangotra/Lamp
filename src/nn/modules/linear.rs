use core::fmt;
use std::{cell::RefCell, rc::Rc};

use crate::nn::parameter::Parameter;

use super::{Module, ModuleParams};

use serde::{Serialize, Serializer, ser::SerializeStruct, Deserialize, Deserializer};
use serde::de::{self, Visitor, MapAccess};

pub struct Linear {
    weight: Rc<RefCell<Parameter>>,
    bias: Rc<RefCell<Parameter>>,
}

impl Serialize for Linear {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let weight: Parameter = self.weight.borrow().to_owned().into();
        let bias: Parameter = self.bias.borrow().to_owned().into();
        let mut state = serializer.serialize_struct("Linear", 2)?;
        state.serialize_field("weight", &weight)?;
        state.serialize_field("bias", &bias)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for Linear {
    fn deserialize<D>(deserializer: D) -> Result<Linear, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field { Weight, Bias }

        struct LinearVisitor;

        impl<'de> Visitor<'de> for LinearVisitor {
            type Value = Linear;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Linear")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Linear, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut weight = None;
                let mut bias = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Weight => {
                            if weight.is_some() {
                                return Err(de::Error::duplicate_field("weight"));
                            }
                            weight = Some(map.next_value()?);
                        }
                        Field::Bias => {
                            if bias.is_some() {
                                return Err(de::Error::duplicate_field("bias"));
                            }
                            bias = Some(map.next_value()?);
                        }
                    }
                }

                let weight = weight.ok_or_else(|| de::Error::missing_field("weight"))?;
                let bias = bias.ok_or_else(|| de::Error::missing_field("bias"))?;

                Ok(Linear {
                    weight: Rc::new(RefCell::new(weight)),
                    bias: Rc::new(RefCell::new(bias)),
                })
            }
        }

        const FIELDS: &'static [&'static str] = &["weight", "bias"];
        deserializer.deserialize_struct("Linear", FIELDS, LinearVisitor)
    }
}

impl Linear {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        let weight = Rc::new(RefCell::new(Parameter::new(&[output_dim, input_dim])));
        let bias = Rc::new(RefCell::new(Parameter::new(&[output_dim, 1])));
        Self {
            weight,
            bias,
        }
    }
}

impl Module for Linear {

    fn forward(&self, input: &Parameter) -> Parameter {
        let weight = self.weight.borrow();
        let bias = self.bias.borrow();

        let output: Parameter = weight.mm(input).into();
        let output = output.add_tensor(&bias);
        output.into()
    }
}

impl ModuleParams for Linear {
    fn parameters(&self) -> Vec<Rc<RefCell<Parameter>>> {
        vec![
            self.weight.clone(),
            self.bias.clone(),
        ]
    }
}
