mod sample_deque;
mod sample_vec;

pub use sample_deque::SampleDeque;
pub use sample_vec::SampleVec;

/// Indicates the action taken by sampling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SamplingOutcome<T> {
    /// The item was selected for sampling, but no item was evicted.
    Selected,

    /// The candidate was selected, the evicted item is returned.
    Replaced(T),

    /// The candidate was rejected and not added to the sample, the candiate is returned
    Rejected(T),
}
