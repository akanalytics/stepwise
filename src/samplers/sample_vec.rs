use super::SamplingOutcome;
use crate::rng::TinyRng;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SampleVec<T> {
    inner: Vec<T>,
    sample_size: usize,
    seen: usize,
    rng: TinyRng,
}

/// Source: <https://en.wikipedia.org/wiki/Reservoir_sampling>
///
/// ```pascal
/// (* S has items to sample, R will contain the result *)
/// ReservoirSample(S[1..n], R[1..k])
///   // fill the reservoir array
///   for i := 1 to k
///       R[i] := S[i]
///   end
///
///   // replace elements with gradually decreasing probability
///   for i := k+1 to n
///     (* randomInteger(a, b) generates a uniform integer from the inclusive range {a, ..., b} *)
///     j := randomInteger(1, i)
///     if j <= k
///         R[j] := S[i]
///     end
///   end
/// end
/// ```
///
impl<T> SampleVec<T> {
    /// Observe a value and possibly store it - *O(1)*.
    ///
    /// Performs a sampling "step", and either
    /// - consuming the value and storing it into the buffer [`SamplingOutcome::Selected`]
    /// - replacing an existing value in the buffer with the new value [`SamplingOutcome::Replaced`]
    /// - returning it back if it's discarded due to the sampling rate [`SamplingOutcome::Rejected`]
    pub fn sample(&mut self, item: T) -> SamplingOutcome<T> {
        self.seen += 1;

        if self.seen <= self.sample_size {
            self.inner.push(item);
            return SamplingOutcome::Selected;
        }
        let j = self.rng.gen_range(self.seen);
        if j < self.sample_size {
            let old = std::mem::replace(&mut self.inner[j], item);
            SamplingOutcome::Replaced(old)
        } else {
            SamplingOutcome::Rejected(item)
        }
    }

    /// Create a new reservoir sampler with the given sample size and seed.
    /// Use usize::MAX for unlimited sampling (taking every items). Note that in unlimited
    /// sampling the Self[`Self::as_unordered_slice`] and associated iterators will always we sorted in the
    /// same order as the input.
    /// The seed is used to initialize the random number generator, and can be used to
    /// makes the sampling determininistic and reproducible.
    /// Use SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos() to get a
    /// random seed.
    ///
    pub fn with_size(sample_size: usize, seed: u64) -> Self {
        let rng = TinyRng::new(seed);
        Self {
            sample_size,
            inner: Vec::new(),
            seen: 0,
            rng,
        }
    }

    /// Count of items in the reservoir.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Empty check
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Consume self and return the internal item buffer.
    /// For unbounded sampling where every item is collected, this will infact be ordered by the
    /// order of the input.
    pub fn into_unordered_vec(self) -> Vec<T> {
        self.inner
    }

    /// Get a view into the collected samples.
    /// For unbounded sampling where every item is collected, this will infact be ordered by the
    /// order of the input.
    pub fn as_unordered_slice(&self) -> &[T] {
        &self.inner
    }

    /// This is irreversible and consumes the sample.
    pub fn into_ordered_vec(mut self) -> Vec<T>
    where
        T: Ord,
    {
        self.inner.sort();
        self.inner
    }

    /// Returns the total number of samples observed (but not necessarily collected) by the
    /// sampler.
    pub fn samples_seen(&self) -> usize {
        self.seen
    }
}
