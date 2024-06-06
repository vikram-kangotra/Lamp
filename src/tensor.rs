use std::{cell::RefCell, fmt::{Display, Formatter}, ops::{Add, Div, Mul, Neg, Sub}, rc::Rc, sync::Arc};

use cudarc::{driver::{CudaDevice, CudaSlice, DeviceRepr, DriverError, LaunchAsync, LaunchConfig, ValidAsZeroBits}, nvrtc::compile_ptx};

use crate::{autograd::{add::AddBackward, div::DivBackward, mul::MulBackward, neg::NegBackward, sub::SubBackward, sum::SumBackward, AutogradOps}, cuda_kernel::{ARITHMETIC_SRC, UTILS_SRC}};

pub trait TensorElement: Copy + Clone + Display + DeviceRepr + ValidAsZeroBits + Into<f32> + Into<f64> + From<f32> + Neg<Output = Self> + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> {}

impl<T> TensorElement for T where T: Copy + Clone + Display + DeviceRepr + ValidAsZeroBits + Into<f32> + Into<f64> + From<f32> + Neg<Output = Self> + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> {}

#[derive(Debug, Clone)]
enum Device<T: TensorElement> {
    CPU(Vec<T>),
    GPU(CudaSlice<T>, Arc<CudaDevice>),
}

impl Display for Device<f32> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Device::CPU(data) => write!(f, "CPU({:?})", data),
            Device::GPU(_, _) => write!(f, "GPU({:?})", self.into_cpu()),
        }
    }
}

impl<T: TensorElement> Device<T> {
    fn into_cpu(&self) -> Vec<T> {
        match self {
            Device::CPU(data) => data.clone(),
            Device::GPU(data, device) => device.dtoh_sync_copy(data).unwrap(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorData<T: TensorElement> {
    shape: Vec<usize>,
    size: usize,
    strides: Vec<usize>,
    device: Device<T>,
    requires_grad: bool,
    grad: Option<Tensor<T>>,
    grad_fn: Option<Rc<AutogradOps<T>>>,
}

#[derive(Debug, Clone)]
pub struct Tensor<T: TensorElement> {
    data: Rc<RefCell<TensorData<T>>>,
}

impl<T: TensorElement> Tensor<T> {

    fn gpu_device(ordinal: usize) -> Arc<CudaDevice> {
        static mut DEVICE: Option<Arc<CudaDevice>> = None;

        unsafe {
            match DEVICE {
                Some(ref device) => device.clone(),
                None => {
                    let device = CudaDevice::new(ordinal).unwrap();
                    Self::initialize_device(&device);
                    DEVICE = Some(device.clone());
                    device
                }
            }
        }
    }

    fn initialize_device(device: &Arc<CudaDevice>) {
        let ptx = compile_ptx(ARITHMETIC_SRC).expect("Failed to compile PTX");
        device.load_ptx(ptx, "arithmetic", &["neg", "add", "add_scalar", "sub", "sub_scalar", "mul", "mul_scalar", "div", "div_scalar", "sum"]).expect("Failed to load PTX");

        let ptx = compile_ptx(UTILS_SRC).expect("Failed to compile PTX");
        device.load_ptx(ptx, "util", &["get_flat_item"]).expect("Failed to load PTX");
    }
    
    fn new_cpu_data(data: &[T], shape: &[usize], requires_grad: bool) -> TensorData<T> {
        TensorData {
            shape: shape.to_vec(),
            size: shape.iter().product(),
            strides: Self::compute_strides(shape),
            device: Device::CPU(data.to_vec()),
            requires_grad,
            grad: None,
            grad_fn: None,
        }
    }

    fn new_gpu_data(data: CudaSlice<T>, device: &Arc<CudaDevice>, shape: &[usize], requires_grad: bool) -> TensorData<T> {
        TensorData {
            shape: shape.to_vec(),
            size: shape.iter().product(),
            strides: Self::compute_strides(shape),
            device: Device::GPU(data, device.clone()),
            requires_grad,
            grad: None,
            grad_fn: None,
        }
    }

    pub fn new(data: &[T], shape: &[usize], requires_grad: bool) -> Self {
        let tensor = Self::new_cpu_data(data, shape, requires_grad);
        Self {
            data: Rc::new(RefCell::new(tensor)),
        }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut stride = 1;
        let mut strides = vec![0; shape.len()];

        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }

        strides
    }

    pub fn get_data(&self) -> Vec<T> {
        self.data.borrow().device.into_cpu()
    }

    pub fn get_shape(&self) -> Vec<usize> {
        self.data.borrow().shape.clone()
    }
    
    pub fn to_gpu(&mut self) -> Result<(), DriverError> {

        let mut self_data = self.data.borrow_mut();

        match &self_data.device {
            Device::CPU(data) => {
                let device = Self::gpu_device(0);
                let data = device.htod_sync_copy(data)?;
                self_data.device = Device::GPU(data, device);
            },
            Device::GPU(_, _) => (),
        }

        Ok(())
    }

    pub fn to_cpu(&mut self) -> Result<(), DriverError> {

        let mut self_data = self.data.borrow_mut();

        match &self_data.device {
            Device::GPU(data, device) => {
                let data = device.dtoh_sync_copy(data)?;
                self_data.device = Device::CPU(data);
            },
            Device::CPU(_) => (),
        }

        Ok(())
    }

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        let data = vec![T::from(0.0f32); shape.iter().product()];
        Self::new(&data, shape, requires_grad)
    }
    
    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        let data = vec![T::from(1.0f32); shape.iter().product()];
        Self::new(&data, shape, requires_grad)
    }

    pub fn ones_like(&self) -> Self {
        let self_data = self.data.borrow();
        Self::ones(&self_data.shape, self_data.requires_grad)
    }

    pub fn grad(&self) -> Option<Self> {
        self.data.borrow().grad.clone()
    }

    pub fn backward(&self, gradient: Option<Self>) {

        let mut self_data = self.data.borrow_mut();

        if !self_data.requires_grad {
            return;
        }

        let gradient = match gradient {
            Some(grad) => grad,
            None => {
                let data = vec![T::from(1.0f32); self_data.size];
                let mut grad = Self::new(&data, &self_data.shape, false);
                if let Device::GPU(_, device) = &self_data.device {
                    grad.to_gpu().unwrap();
                }
                grad
            }
        };

        if self_data.grad.is_none() {
            self_data.grad = Some(gradient.clone());
        } else {
            let mut grad = self_data.grad.as_ref().unwrap().clone();
            grad = &grad + &gradient;
            self_data.grad = Some(grad);
        }

        if self_data.grad_fn.is_none() {
            return;
        }

        let grads = self_data.grad_fn.as_ref().unwrap().backward(gradient);

        self_data.grad_fn
            .as_ref()
            .unwrap()
            .inputs()
            .iter()
            .zip(grads.iter())
            .for_each(|(input, grad)| input.backward(Some(grad.clone())));

    }

    pub fn get_flat_item(&self, offset: usize) -> T {

        let self_data = self.data.borrow();

        match &self_data.device {
            Device::CPU(data) => data[offset].clone(),
            Device::GPU(data, device) => self.gpu_get_flat_item(data, device, offset),
        }
    }

    pub fn gpu_get_flat_item(&self, data: &CudaSlice<T>, device: &Arc<CudaDevice>, offset: usize) -> T {
        let mut result = device.alloc_zeros(1).unwrap();

        let cfg = LaunchConfig::for_num_elems(1);

        let func = device.get_func("util", "get_flat_item").unwrap();

        unsafe {
            func.launch(cfg, (data, &mut result, offset)).unwrap();
        }

        let result = device.dtoh_sync_copy(&result).unwrap();

        result[0]
    }

    pub fn neg_tensor(&self) -> Self {

        let result = match &self.data.borrow().device {
            Device::CPU(data) => self.cpu_neg_tensor(data),
            Device::GPU(data, device) => self.gpu_neg_tensor(data, device),
        };

        let result_ref = result.data.clone();
        let mut result_data = result_ref.borrow_mut();

        result_data.requires_grad = self.data.borrow().requires_grad;

        if result_data.requires_grad {
            result_data.grad_fn = Some(Rc::new(AutogradOps::NegBackward(NegBackward::new([self.data.clone()]))));
        }

        result
    }

    fn cpu_neg_tensor(&self, data: &Vec<T>) -> Self {

        let self_data = self.data.borrow();

        let data = data.iter().map(|a| -a.clone()).collect::<Vec<T>>();
        Self::new(&data, &self_data.shape, false)
    }

    fn gpu_neg_tensor(&self, data: &CudaSlice<T>, device: &Arc<CudaDevice>) -> Self {

        let self_data = self.data.borrow();

        let mut result = device.alloc_zeros(self_data.size).unwrap();

        let cfg = LaunchConfig::for_num_elems(self_data.size as u32);

        let func = device.get_func("arithmetic", "neg").unwrap();

        unsafe {
            func.launch(cfg, (data, &mut result, self_data.size)).unwrap();
        }

        Self {
            data: Rc::new(RefCell::new(Self::new_gpu_data(result, device, &self_data.shape, self_data.requires_grad)))
        }
    }

    pub fn add_tensor(&self, other: &Self) -> Self {

        let self_data = self.data.borrow();
        let other_data = other.data.borrow();

        assert_eq!(self_data.shape, other_data.shape, "Cannot add tensors with different shapes");

        let result = match (&self_data.device, &other_data.device) {
            (Device::CPU(data), Device::CPU(other_data)) => self.cpu_add_tensor(data, &other_data),
            (Device::GPU(data, device), Device::GPU(other_data, other_device)) => self.gpu_add_tensor(data, other_data, device),
            _ => panic!("Cannot add CPU tensor to GPU tensor or vice versa. Move tensors to the same device first."),
        };

        let result_ref = result.data.clone();
        let mut result_data = result_ref.borrow_mut();

        result_data.requires_grad = self_data.requires_grad || other_data.requires_grad; 

        if result_data.requires_grad {
            result_data.grad_fn = Some(Rc::new(AutogradOps::AddBackward(AddBackward::new([self.data.clone(), other.data.clone()]))));
        }

        result
    }

    fn cpu_add_tensor(&self, data: &Vec<T>, other: &Vec<T>) -> Self {

        let self_data = self.data.borrow();

        let data = data.iter().zip(other.iter()).map(|(a, b)| a.clone() + b.clone()).collect::<Vec<T>>();
        Self::new(&data, &self_data.shape, false)
    }

    fn gpu_add_tensor(&self, data: &CudaSlice<T>, other: &CudaSlice<T>, device: &Arc<CudaDevice>) -> Self {

        let self_data = self.data.borrow();

        let mut result = device.alloc_zeros(self_data.size).unwrap();

        let cfg = LaunchConfig::for_num_elems(self_data.size as u32);

        let func = device.get_func("arithmetic", "add").unwrap();

        unsafe {
            func.launch(cfg, (data, other, &mut result, self_data.size)).unwrap();
        }

        Self {
            data: Rc::new(RefCell::new(Self::new_gpu_data(result, device, &self_data.shape, self_data.requires_grad)))
        }
    }

    pub fn add_scalar(&self, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let result = match &self_data.device {
            Device::CPU(data) => self.cpu_add_scalar(data, scalar),
            Device::GPU(data, device) => self.gpu_add_scalar(data, device, scalar),
        };

        let result_ref = result.data.clone();
        let mut result_data = result_ref.borrow_mut();

        result_data.requires_grad = self_data.requires_grad;

        if result_data.requires_grad {
            let data = vec![scalar; self_data.size];
            let mut other = Self::new(&data, &self_data.shape, false);
            if let Device::GPU(_, device) = &self_data.device {
                other.to_gpu().unwrap();
            }
            result_data.grad_fn = Some(Rc::new(AutogradOps::AddBackward(AddBackward::new([self.data.clone(), other.data.clone()]))));
        }

        result
    }

    fn cpu_add_scalar(&self, data: &Vec<T>, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let data = data.iter().map(|a| a.clone() + scalar).collect::<Vec<T>>();
        Self::new(&data, &self_data.shape, false)
    }

    fn gpu_add_scalar(&self, data: &CudaSlice<T>, device: &Arc<CudaDevice>, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let mut result = device.alloc_zeros(self_data.size).unwrap();

        let cfg = LaunchConfig::for_num_elems(self_data.size as u32);

        let func = device.get_func("arithmetic", "add_scalar").unwrap();

        unsafe {
            func.launch(cfg, (data, scalar, &mut result, self_data.size)).unwrap();
        }

        Self {
            data: Rc::new(RefCell::new(Self::new_gpu_data(result, device, &self_data.shape, self_data.requires_grad)))
        }
    }

    pub fn sub_tensor(&self, other: &Self) -> Self {

        let self_data = self.data.borrow();
        let other_data = other.data.borrow();

        assert_eq!(self_data.shape, other_data.shape, "Cannot subtract tensors with different shapes");

        let result = match (&self_data.device, &other_data.device) {
            (Device::CPU(data), Device::CPU(other_data)) => self.cpu_sub_tensor(data, &other_data),
            (Device::GPU(data, device), Device::GPU(other_data, other_device)) => self.gpu_sub_tensor(data, other_data, device),
            _ => panic!("Cannot subtract CPU tensor from GPU tensor or vice versa. Move tensors to the same device first."),
        };

        let result_ref = result.data.clone();
        let mut result_data = result_ref.borrow_mut();

        result_data.requires_grad = self_data.requires_grad || other_data.requires_grad; 

        if result_data.requires_grad {
            result_data.grad_fn = Some(Rc::new(AutogradOps::SubBackward(SubBackward::new([self.data.clone(), other.data.clone()]))));
        }

        result
    }

    fn cpu_sub_tensor(&self, data: &Vec<T>, other: &Vec<T>) -> Self {

        let self_data = self.data.borrow();

        let data = data.iter().zip(other.iter()).map(|(a, b)| a.clone() - b.clone()).collect::<Vec<T>>();
        Self::new(&data, &self_data.shape, false)
    }

    fn gpu_sub_tensor(&self, data: &CudaSlice<T>, other: &CudaSlice<T>, device: &Arc<CudaDevice>) -> Self {

        let self_data = self.data.borrow();

        let mut result = device.alloc_zeros(self_data.size).unwrap();

        let cfg = LaunchConfig::for_num_elems(self_data.size as u32);

        let func = device.get_func("arithmetic", "sub").unwrap();

        unsafe {
            func.launch(cfg, (data, other, &mut result, self_data.size)).unwrap();
        }

        Self {
            data: Rc::new(RefCell::new(Self::new_gpu_data(result, device, &self_data.shape, self_data.requires_grad)))
        }
    }

    pub fn sub_scalar(&self, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let result = match &self_data.device {
            Device::CPU(data) => self.cpu_sub_scalar(data, scalar),
            Device::GPU(data, device) => self.gpu_sub_scalar(data, device, scalar),
        };

        let result_ref = result.data.clone();
        let mut result_data = result_ref.borrow_mut();

        result_data.requires_grad = self_data.requires_grad;

        if result_data.requires_grad {
            let data = vec![scalar; self_data.size];
            let mut other = Self::new(&data, &self_data.shape, false);
            if let Device::GPU(_, device) = &self_data.device {
                other.to_gpu().unwrap();
            }
            result_data.grad_fn = Some(Rc::new(AutogradOps::SubBackward(SubBackward::new([self.data.clone(), other.data.clone()]))));
        }

        result
    }

    fn cpu_sub_scalar(&self, data: &Vec<T>, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let data = data.iter().map(|a| a.clone() - scalar).collect::<Vec<T>>();
        Self::new(&data, &self_data.shape, false)
    }

    fn gpu_sub_scalar(&self, data: &CudaSlice<T>, device: &Arc<CudaDevice>, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let mut result = device.alloc_zeros(self_data.size).unwrap();

        let cfg = LaunchConfig::for_num_elems(self_data.size as u32);

        let func = device.get_func("arithmetic", "sub_scalar").unwrap();

        unsafe {
            func.launch(cfg, (data, scalar, &mut result, self_data.size)).unwrap();
        }

        Self {
            data: Rc::new(RefCell::new(Self::new_gpu_data(result, device, &self_data.shape, self_data.requires_grad)))
        }
    }

    pub fn sub_from_scalar(&self, scalar: T) -> Self {
        self.neg_tensor().add_scalar(scalar)
    }

    pub fn mul_tensor(&self, other: &Self) -> Self {

        let self_data = self.data.borrow();
        let other_data = other.data.borrow();

        assert_eq!(self_data.shape, other_data.shape, "Cannot multiply tensors with different shapes");

        let result = match (&self_data.device, &other_data.device) {
            (Device::CPU(data), Device::CPU(other_data)) => self.cpu_mul_tensor(data, &other_data),
            (Device::GPU(data, device), Device::GPU(other_data, other_device)) => self.gpu_mul_tensor(data, other_data, device),
            _ => panic!("Cannot multiply CPU tensor with GPU tensor or vice versa. Move tensors to the same device first."),
        };

        let result_ref = result.data.clone();
        let mut result_data = result_ref.borrow_mut();

        result_data.requires_grad = self_data.requires_grad || other_data.requires_grad; 

        if result_data.requires_grad {
            result_data.grad_fn = Some(Rc::new(AutogradOps::MulBackward(MulBackward::new([self.data.clone(), other.data.clone()]))));
        }

        result
    }

    fn cpu_mul_tensor(&self, data: &Vec<T>, other: &Vec<T>) -> Self {

        let self_data = self.data.borrow();

        let data = data.iter().zip(other.iter()).map(|(a, b)| *a * *b).collect::<Vec<T>>();
        Self::new(&data, &self_data.shape, false)
    }

    fn gpu_mul_tensor(&self, data: &CudaSlice<T>, other: &CudaSlice<T>, device: &Arc<CudaDevice>) -> Self {

        let self_data = self.data.borrow();

        let mut result = device.alloc_zeros(self_data.size).unwrap();

        let cfg = LaunchConfig::for_num_elems(self_data.size as u32);

        let func = device.get_func("arithmetic", "mul").unwrap();

        unsafe {
            func.launch(cfg, (data, other, &mut result, self_data.size)).unwrap();
        }

        Self {
            data: Rc::new(RefCell::new(Self::new_gpu_data(result, device, &self_data.shape, self_data.requires_grad)))
        }
    }

    pub fn mul_scalar(&self, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let result = match &self_data.device {
            Device::CPU(data) => self.cpu_mul_scalar(data, scalar),
            Device::GPU(data, device) => self.gpu_mul_scalar(data, device, scalar),
        };

        let result_ref = result.data.clone();
        let mut result_data = result_ref.borrow_mut();

        result_data.requires_grad = self_data.requires_grad;

        if result_data.requires_grad {
            let data = vec![scalar; self_data.size];
            let mut other = Self::new(&data, &self_data.shape, false);
            if let Device::GPU(_, device) = &self_data.device {
                other.to_gpu().unwrap();
            }
            result_data.grad_fn = Some(Rc::new(AutogradOps::MulBackward(MulBackward::new([self.data.clone(), other.data.clone()]))));
        }

        result
    }

    fn cpu_mul_scalar(&self, data: &Vec<T>, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let data = data.iter().map(|a| *a * scalar).collect::<Vec<T>>();
        Self::new(&data, &self_data.shape, false)
    }

    fn gpu_mul_scalar(&self, data: &CudaSlice<T>, device: &Arc<CudaDevice>, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let mut result = device.alloc_zeros(self_data.size).unwrap();

        let cfg = LaunchConfig::for_num_elems(self_data.size as u32);

        let func = device.get_func("arithmetic", "mul_scalar").unwrap();

        unsafe {
            func.launch(cfg, (data, scalar, &mut result, self_data.size)).unwrap();
        }

        Self {
            data: Rc::new(RefCell::new(Self::new_gpu_data(result, device, &self_data.shape, self_data.requires_grad)))
        }
    }
    
    fn div_tensor(&self, other: &Self) -> Self {

        let self_data = self.data.borrow();
        let other_data = other.data.borrow();

        assert_eq!(self_data.shape, other_data.shape, "Cannot divide tensors with different shapes");

        let result = match (&self_data.device, &other_data.device) {
            (Device::CPU(data), Device::CPU(other_data)) => self.cpu_div_tensor(data, &other_data),
            (Device::GPU(data, device), Device::GPU(other_data, other_device)) => self.gpu_div_tensor(data, other_data, device),
            _ => panic!("Cannot divide CPU tensor by GPU tensor or vice versa. Move tensors to the same device first."),
        };

        let result_ref = result.data.clone();
        let mut result_data = result_ref.borrow_mut();

        result_data.requires_grad = self_data.requires_grad || other_data.requires_grad; 

        if result_data.requires_grad {
            result_data.grad_fn = Some(Rc::new(AutogradOps::DivBackward(DivBackward::new([self.data.clone(), other.data.clone()]))));
        }

        result
    }

    fn cpu_div_tensor(&self, data: &Vec<T>, other: &Vec<T>) -> Self {

        let self_data = self.data.borrow();

        let data = data.iter().zip(other.iter()).map(|(a, b)| *a / *b).collect::<Vec<T>>();
        Self::new(&data, &self_data.shape, false)
    }

    fn gpu_div_tensor(&self, data: &CudaSlice<T>, other: &CudaSlice<T>, device: &Arc<CudaDevice>) -> Self {

        let self_data = self.data.borrow();

        let mut result = device.alloc_zeros(self_data.size).unwrap();

        let cfg = LaunchConfig::for_num_elems(self_data.size as u32);

        let func = device.get_func("arithmetic", "div").unwrap();

        unsafe {
            func.launch(cfg, (data, other, &mut result, self_data.size)).unwrap();
        }

        Self {
            data: Rc::new(RefCell::new(Self::new_gpu_data(result, device, &self_data.shape, self_data.requires_grad)))
        }
    }
 
    pub fn div_scalar(&self, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let result = match &self_data.device {
            Device::CPU(data) => self.cpu_div_scalar(data, scalar),
            Device::GPU(data, device) => self.gpu_div_scalar(data, device, scalar),
        };

        let result_ref = result.data.clone();
        let mut result_data = result_ref.borrow_mut();

        result_data.requires_grad = self_data.requires_grad;

        if result_data.requires_grad {
            let data = vec![scalar; self_data.size];
            let mut other = Self::new(&data, &self_data.shape, false);
            if let Device::GPU(_, device) = &self_data.device {
                other.to_gpu().unwrap();
            }
            result_data.grad_fn = Some(Rc::new(AutogradOps::DivBackward(DivBackward::new([self.data.clone(), other.data.clone()]))));
        }

        result
    }

    fn cpu_div_scalar(&self, data: &Vec<T>, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let data = data.iter().map(|a| *a / scalar).collect::<Vec<T>>();
        Self::new(&data, &self_data.shape, false)
    }

    fn gpu_div_scalar(&self, data: &CudaSlice<T>, device: &Arc<CudaDevice>, scalar: T) -> Self {

        let self_data = self.data.borrow();

        let mut result = device.alloc_zeros(self_data.size).unwrap();

        let cfg = LaunchConfig::for_num_elems(self_data.size as u32);

        let func = device.get_func("arithmetic", "div_scalar").unwrap();

        unsafe {
            func.launch(cfg, (data, scalar, &mut result, self_data.size)).unwrap();
        }

        Self {
            data: Rc::new(RefCell::new(Self::new_gpu_data(result, device, &self_data.shape, self_data.requires_grad)))
        }
    }

    pub fn div_from_scalar(&self, scalar: T) -> Self {
        self.reciprocal().mul_scalar(scalar)
    }

    pub fn reciprocal(&self) -> Self {
        let mut this = Self::ones(&self.get_shape(), self.data.borrow().requires_grad);
        if let Device::GPU(_, device) = &self.data.borrow().device {
            this.to_gpu().unwrap();
        }
        this.div_tensor(self)
    }

    pub fn sum(&self) -> Self {
        let self_data = self.data.borrow();

        let result = match &self_data.device {
            Device::CPU(data) => self.cpu_sum(data),
            Device::GPU(data, device) => self.gpu_sum(data, device),
        };

        let result_ref = result.data.clone();
        let mut result_data = result_ref.borrow_mut();

        result_data.requires_grad = self_data.requires_grad;

        if result_data.requires_grad {
            result_data.grad_fn = Some(Rc::new(AutogradOps::SumBackward(SumBackward::new([self.data.clone()]))));
        }

        result
    }

    fn cpu_sum(&self, data: &Vec<T>) -> Self {

        let self_data = self.data.borrow();

        let sum = data.iter().fold(T::from(0.0f32), |acc, x| acc + *x);
        let data = vec![sum; 1];
        Self::new(&data, &[1], false)
    }

    fn gpu_sum(&self, data: &CudaSlice<T>, device: &Arc<CudaDevice>) -> Self {

        let self_data = self.data.borrow();

        let mut result = device.alloc_zeros(1).unwrap();

        let mut cfg = LaunchConfig::for_num_elems(self_data.size as u32);
        cfg.shared_mem_bytes = 1024 * std::mem::size_of::<T>() as u32;

        let func = device.get_func("arithmetic", "sum").unwrap();

        unsafe {
            func.launch(cfg, (data, &mut result, self_data.size)).unwrap();
        }

        Self {
            data: Rc::new(RefCell::new(Self::new_gpu_data(result, device, &[1], self_data.requires_grad)))
        }
    }
}

pub trait FromRcRefCell<T> {
    fn from_rc_refcell(data: &Rc<RefCell<T>>) -> Self;
}

impl<T: TensorElement> FromRcRefCell<TensorData<T>> for Tensor<T> {
    fn from_rc_refcell(data: &Rc<RefCell<TensorData<T>>>) -> Self {
        Self {
            data: data.clone()
        }
    }
}

impl<T: TensorElement> Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(")?;
        self.print(f, 0, 0)?;
        write!(f, ")")
    }
}

impl<T: TensorElement> Tensor<T> {
    fn print(&self, f: &mut Formatter<'_>, index: usize, offset: usize) -> std::fmt::Result {

        let self_data = self.data.borrow();

        if index == self_data.shape.len() {
            write!(f, "{:.2}", self.get_flat_item(offset))
        } else {
            write!(f, "[")?;
            for i in 0..self_data.shape[index] {
                self.print(f, index + 1, offset + i * self_data.strides[index])?;
                if i < self_data.shape[index] - 1 {
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")
        }
    }
}

impl<T: TensorElement> Neg for &Tensor<T> {
    type Output = Tensor<T>;

    fn neg(self) -> Self::Output {
        self.neg_tensor()
    }
}

impl<T: TensorElement> Add<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, other: &Tensor<T>) -> Self::Output {
        self.add_tensor(other)
    }
}

impl<T: TensorElement> Add<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn add(self, other: T) -> Self::Output {
        self.add_scalar(other)
    }
}

impl<T: TensorElement> Sub<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, other: &Tensor<T>) -> Self::Output {
        self.sub_tensor(other)
    }
}

impl<T: TensorElement> Sub<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn sub(self, other: T) -> Self::Output {
        self.sub_scalar(other)
    }
}

impl<T: TensorElement> Mul<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: &Tensor<T>) -> Self::Output {
        self.mul_tensor(other)
    }
}

impl<T: TensorElement> Mul<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn mul(self, other: T) -> Self::Output {
        self.mul_scalar(other)
    }
}

impl<T: TensorElement> Div<&Tensor<T>> for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, other: &Tensor<T>) -> Self::Output {
        self.div_tensor(other)
    }
}

impl<T: TensorElement> Div<T> for &Tensor<T> {
    type Output = Tensor<T>;

    fn div(self, other: T) -> Self::Output {
        self.div_scalar(other)
    }
}

#[cfg(test)]
mod test {
    use crate::tensor::Tensor;

    #[test]
    fn test_neg_tensor() {

        let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = -&a;
        b.backward(None);

        assert_eq!(b.get_data(), &[-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_gpu_neg_tensor() {

        let mut a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        a.to_gpu().unwrap();
        let b = -&a;
        b.backward(None);

        assert_eq!(b.get_data(), &[-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_add_tensor() {

        let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = Tensor::<f32>::new(&[4.0, 3.0, 2.0, 1.0], &[2, 2], true);
        let c = &a + &b;
        c.backward(None);

        assert_eq!(c.get_data(), &[5.0, 5.0, 5.0, 5.0]);
        assert_eq!(c.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(b.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_gpu_add_tensor() {

        let mut a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let mut b = Tensor::<f32>::new(&[4.0, 3.0, 2.0, 1.0], &[2, 2], true);
        a.to_gpu().unwrap();
        b.to_gpu().unwrap();
        let c = &a + &b;
        c.backward(None);

        assert_eq!(c.get_data(), &[5.0, 5.0, 5.0, 5.0]);
        assert_eq!(c.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(b.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_add_scalar() {

        let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = &a + 5.0;
        b.backward(None);

        assert_eq!(b.get_data(), &[6.0, 7.0, 8.0, 9.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_gpu_add_scalar() {

        let mut a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        a.to_gpu().unwrap();
        let b = &a + 5.0;
        b.backward(None);

        assert_eq!(b.get_data(), &[6.0, 7.0, 8.0, 9.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sub_tensor() {

        let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = Tensor::<f32>::new(&[4.0, 3.0, 2.0, 1.0], &[2, 2], true);
        let c = &a - &b;
        c.backward(None);

        assert_eq!(c.get_data(), &[-3.0, -1.0, 1.0, 3.0]);
        assert_eq!(c.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(b.grad().unwrap().get_data(), &[-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_gpu_sub_tensor() {

        let mut a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let mut b = Tensor::<f32>::new(&[4.0, 3.0, 2.0, 1.0], &[2, 2], true);
        a.to_gpu().unwrap();
        b.to_gpu().unwrap();
        let c = &a - &b;
        c.backward(None);

        assert_eq!(c.get_data(), &[-3.0, -1.0, 1.0, 3.0]);
        assert_eq!(c.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(b.grad().unwrap().get_data(), &[-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_sub_scalar() {

        let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = &a - 5.0;
        b.backward(None);

        assert_eq!(b.get_data(), &[-4.0, -3.0, -2.0, -1.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_gpu_sub_scalar() {

        let mut a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        a.to_gpu().unwrap();
        let b = &a - 5.0;
        b.backward(None);

        assert_eq!(b.get_data(), &[-4.0, -3.0, -2.0, -1.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sub_from_scalar() {

        let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = a.sub_from_scalar(5.0);
        b.backward(None);

        assert_eq!(b.get_data(), &[4.0, 3.0, 2.0, 1.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[-1.0, -1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_mul_tensor() {

        let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = Tensor::<f32>::new(&[4.0, 3.0, 2.0, 1.0], &[2, 2], true);
        let c = &a * &b;
        c.backward(None);

        assert_eq!(c.get_data(), &[4.0, 6.0, 6.0, 4.0]);
        assert_eq!(c.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[4.0, 3.0, 2.0, 1.0]);
        assert_eq!(b.grad().unwrap().get_data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gpu_mul_tensor() {

        let mut a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 3.0, 2.0, 1.0], &[2, 3], true);
        let mut b = Tensor::<f32>::new(&[3.0, 2.0, 1.0, 1.0, 2.0, 3.0], &[2, 3], true);
        a.to_gpu().unwrap();
        b.to_gpu().unwrap();
        let c = &a * &b;
        c.backward(None);

        assert_eq!(c.get_data(), &[3.0, 4.0, 3.0, 3.0, 4.0, 3.0]);
        assert_eq!(c.get_shape(), &[2, 3]);
        assert_eq!(a.grad().unwrap().get_data(), &[3.0, 2.0, 1.0, 1.0, 2.0, 3.0]);
        assert_eq!(b.grad().unwrap().get_data(), &[1.0, 2.0, 3.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_mul_scalar() {

        let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = &a * 5.0;
        b.backward(None);

        assert_eq!(b.get_data(), &[5.0, 10.0, 15.0, 20.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_gpu_mul_scalar() {

        let mut a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 3.0, 2.0, 1.0], &[2, 3], true);
        a.to_gpu().unwrap();
        let b = &a * 5.0;
        b.backward(None);

        assert_eq!(b.get_data(), &[5.0, 10.0, 15.0, 15.0, 10.0, 5.0]);
        assert_eq!(b.get_shape(), &[2, 3]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 3]);
        assert_eq!(a.grad().unwrap().get_data(), &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
    }

    #[test]
    fn test_div_tensor() {

        let a = Tensor::<f32>::new(&[4.0, 6.0, 6.0, 4.0], &[2, 2], true);
        let b = Tensor::<f32>::new(&[4.0, 3.0, 2.0, 1.0], &[2, 2], true);
        let c = &a / &b;
        c.backward(None);

        assert_eq!(c.get_data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(c.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 1.0 / 1.0]);
        assert_eq!(b.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(b.grad().unwrap().get_data(), &[-4.0 / (4.0 * 4.0), -6.0 / (3.0 * 3.0), -6.0 / (2.0 * 2.0), -4.0 / (1.0 * 1.0)]);
    }

    #[test]
    fn test_gpu_div_tensor() {

        let mut a = Tensor::<f32>::new(&[3.0, 4.0, 3.0, 3.0, 4.0, 3.0], &[2, 3], true);
        let mut b = Tensor::<f32>::new(&[3.0, 2.0, 1.0, 1.0, 2.0, 3.0], &[2, 3], true);
        a.to_gpu().unwrap();
        b.to_gpu().unwrap();
        let c = &a / &b;
        c.backward(None);

        assert_eq!(c.get_data(), &[1.0, 2.0, 3.0, 3.0, 2.0, 1.0]);
        assert_eq!(c.get_shape(), &[2, 3]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 3]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0 / 3.0, 1.0 / 2.0, 1.0 / 1.0, 1.0 / 1.0, 1.0 / 2.0, 1.0 / 3.0]);
        assert_eq!(b.grad().unwrap().get_shape(), &[2, 3]);
        assert_eq!(b.grad().unwrap().get_data(), &[-3.0 / (3.0 * 3.0), -4.0 / (2.0 * 2.0), -3.0 / (1.0 * 1.0), -3.0 / (1.0 * 1.0), -4.0 / (2.0 * 2.0), -3.0 / (3.0 * 3.0)]);
    }

    #[test]
    fn test_div_scalar() {

        let a = Tensor::<f32>::new(&[5.0, 10.0, 15.0, 20.0], &[2, 2], true);
        let b = &a / 5.0;
        b.backward(None);

        assert_eq!(b.get_data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0]);
    }

    #[test]
    fn test_gpu_div_scalar() {

        let mut a = Tensor::<f32>::new(&[5.0, 10.0, 15.0, 15.0, 10.0, 5.0], &[2, 3], true);
        a.to_gpu().unwrap();
        let b = &a / 5.0;
        b.backward(None);

        assert_eq!(b.get_data(), &[1.0, 2.0, 3.0, 3.0, 2.0, 1.0]);
        assert_eq!(b.get_shape(), &[2, 3]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 3]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0, 1.0 / 5.0]);
    }

    #[test]
    fn test_div_from_scalar() {

        let a = Tensor::<f32>::new(&[4.0, 3.0, 2.0, 1.0], &[2, 2], true);
        let b = a.div_from_scalar(2.0);
        b.backward(None);

        assert_eq!(b.get_data(), &[2.0 / 4.0, 2.0 / 3.0, 2.0 / 2.0, 2.0 / 1.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[-2.0 / (4.0 * 4.0), -2.0 / (3.0 * 3.0), -2.0 / (2.0 * 2.0), -2.0 / (1.0 * 1.0)]);
    }

    #[test]
    fn test_gpu_div_from_scalar() {

        let mut a = Tensor::<f32>::new(&[4.0, 3.0, 2.0, 1.0], &[2, 2], true);
        a.to_gpu().unwrap();
        let b = a.div_from_scalar(2.0);
        b.backward(None);

        assert_eq!(b.get_data(), &[2.0 / 4.0, 2.0 / 3.0, 2.0 / 2.0, 2.0 / 1.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[-2.0 / (4.0 * 4.0), -2.0 / (3.0 * 3.0), -2.0 / (2.0 * 2.0), -2.0 / (1.0 * 1.0)]);
    }

    #[test]
    fn test_reciprocal() {

        let a = Tensor::<f32>::new(&[4.0, 3.0, 2.0, 1.0], &[2, 2], true);
        let b = a.reciprocal();
        b.backward(None);

        assert_eq!(b.get_data(), &[1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 1.0 / 1.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[-1.0 / (4.0 * 4.0), -1.0 / (3.0 * 3.0), -1.0 / (2.0 * 2.0), -1.0 / (1.0 * 1.0)]);
    }

    #[test]
    fn test_gpu_reciprocal() {

        let mut a = Tensor::<f32>::new(&[4.0, 3.0, 2.0, 1.0], &[2, 2], true);
        a.to_gpu().unwrap();
        let b = a.reciprocal();
        a.to_gpu().unwrap();
        b.backward(None);

        assert_eq!(b.get_data(), &[1.0 / 4.0, 1.0 / 3.0, 1.0 / 2.0, 1.0 / 1.0]);
        assert_eq!(b.get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[-1.0 / (4.0 * 4.0), -1.0 / (3.0 * 3.0), -1.0 / (2.0 * 2.0), -1.0 / (1.0 * 1.0)]);
    }

    #[test]
    fn test_sum() {

        let a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = a.sum();
        b.backward(None);

        assert_eq!(b.get_data(), &[10.0]);
        assert_eq!(b.get_shape(), &[1]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_gpu_sum() {

        let mut a = Tensor::<f32>::new(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        a.to_gpu().unwrap();
        let b = a.sum();
        b.backward(None);

        assert_eq!(b.get_data(), &[10.0]);
        assert_eq!(b.get_shape(), &[1]);
        assert_eq!(a.grad().unwrap().get_shape(), &[2, 2]);
        assert_eq!(a.grad().unwrap().get_data(), &[1.0, 1.0, 1.0, 1.0]);
    }
}