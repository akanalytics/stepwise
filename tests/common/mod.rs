use log::trace;
use std::{error::Error, fmt, fmt::Debug, io, marker::PhantomData, ops::ControlFlow};
use stepwise::{Algo, FileFormat};

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CountingAlgo(pub usize);

impl CountingAlgo {
    pub fn x(&self) -> usize {
        self.0
    }
}

pub trait DoubleCount {
    #[allow(dead_code)]
    fn twice_x(&self) -> usize;
}

impl DoubleCount for CountingAlgo {
    fn twice_x(&self) -> usize {
        self.x() * 2
    }
}

impl Algo for CountingAlgo {
    type Error = fmt::Error;

    fn step(&mut self) -> (ControlFlow<()>, Result<(), Self::Error>) {
        trace!("adding 1 to {} (MAX-{})", self.0, usize::MAX - self.0);

        let res = self.0.checked_add(1).ok_or(fmt::Error);
        match res {
            Ok(v) => self.0 = v,
            Err(e) => return (ControlFlow::Break(()), Err(e)),
        };
        (ControlFlow::Continue(()), Ok(()))
    }

    fn read_file(&mut self, fmt: &FileFormat, r: &mut dyn io::Read) -> Result<(), io::Error> {
        match fmt {
            FileFormat::Json => *self = serde_json::from_reader(r)?,
            _ => return Err(fmt.unsupported_error()),
        }
        Ok(())
    }

    fn write_file(&self, fmt: &FileFormat, w: &mut dyn io::Write) -> Result<(), io::Error> {
        match fmt {
            FileFormat::Json => serde_json::to_writer_pretty(w, &self)?,
            FileFormat::Csv { with_header: true } => writeln!(w, "count")?,
            FileFormat::Csv { with_header: false } => writeln!(w, "{}", self.x())?,
            _ => return Err(fmt.unsupported_error()),
        }
        Ok(())
    }
}

pub struct FunctionAlgo<F, A, E>
where
    F: FnMut(A) -> Result<A, E>,
{
    pub f: F,
    pub a: A,
    _phantom: PhantomData<fn(A) -> Result<A, E>>,
}

impl<F, A, E> FunctionAlgo<F, A, E>
where
    F: FnMut(A) -> Result<A, E>,
    A: Copy,
{
    #[allow(dead_code)]
    pub fn from_fn_and_value(f: F, a: A) -> Self {
        Self {
            f,
            a,
            _phantom: PhantomData,
        }
    }

    #[allow(dead_code)]
    pub fn x(&self) -> A {
        self.a
    }
}

impl<F, A, E> Debug for FunctionAlgo<F, A, E>
where
    F: FnMut(A) -> Result<A, E>,
    A: Copy + Debug,
    E: Error + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("FunctionAlgo").field("a", &self.a).finish()
    }
}

impl<F, A, E> Algo for FunctionAlgo<F, A, E>
where
    F: FnMut(A) -> Result<A, E>,
    A: Copy,
    E: std::error::Error + Send + Sync + 'static,
{
    type Error = E;

    fn step(&mut self) -> (ControlFlow<()>, Result<(), Self::Error>) {
        let res = (self.f)(self.a);
        match res {
            Ok(fa) => self.a = fa,
            Err(e) => return (ControlFlow::Break(()), Err(e)),
        };
        (ControlFlow::Continue(()), Ok(()))
    }
}
