use std::{any::Any, error::Error, ops::ControlFlow, sync::Arc};

use crate::Algo;

pub type DynAlgoError = Arc<dyn Error + Send + Sync + 'static>;

pub struct DynAlgo {
    pub(crate) inner: Box<dyn DynAlgoTrait>,
}

impl DynAlgo {
    pub fn new<A>(algo: A) -> Self
    where
        A: Algo + 'static,
    {
        Self {
            inner: Box::new(AlgoWrapper { algo }),
        }
    }
}

// Internal trait for dynamic dispatch
pub(crate) trait DynAlgoTrait {
    fn step(&mut self) -> (ControlFlow<()>, Result<(), DynAlgoError>);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

struct AlgoWrapper<A: Algo + 'static> {
    algo: A,
}

impl<A: Algo> DynAlgoTrait for AlgoWrapper<A> {
    fn step(&mut self) -> (ControlFlow<()>, Result<(), DynAlgoError>) {
        let (cf, result) = self.algo.step();
        let result = result.map_err(|e| Arc::new(e) as DynAlgoError);
        (cf, result)
    }

    fn as_any(&self) -> &dyn Any {
        &self.algo
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.algo
    }
}

// Implement the original trait for DynAlgo using erased error type
impl Algo for DynAlgo {
    type Error = DynAlgoError;

    fn step(&mut self) -> (ControlFlow<()>, Result<(), Self::Error>) {
        self.inner.step()
    }
}

/// DynAlgo result from calling [`Driver::into_dyn`]
///
/// To invoke methods on the underlying algorithm, you will need to downcast to
/// the concrete algorithm type. See [`Driver::into_dyn`] for an example.
///
impl DynAlgo {
    pub fn downcast_mut<A: 'static>(&mut self) -> Option<&mut A> {
        self.inner.as_any_mut().downcast_mut::<A>()
    }

    pub fn downcast_ref<A: 'static>(&self) -> Option<&A> {
        self.inner.as_any().downcast_ref::<A>()
    }
}
