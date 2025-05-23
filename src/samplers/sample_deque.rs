use super::{Sampler, SamplingOutcome};
use crate::samplers::SampleVec;
use std::collections::VecDeque;

const RANDOM_SEED: u64 = 42;

/// A segmented vector that divides elements into three regions: head, middle, and tail.
///
/// The `head` and `tail` regions have fixed capacities, while the `middle` region can
/// either grow dynamically (unbounded) or have a fixed capacity with occasional replacement of items (sparse behavior).
///
/// Typical uses might include sampling from a large stream of data for plotting results.
/// The first and last data items seen are perhaps important to plot, so all the data is retained for
/// these head and tail regions. The middle region is used to sample from the rest of the data, choosing
/// some items but discarding others.
///
/// The collection retains its order of insertion/sampling so that [`into_ordered_vec`](Self::into_ordered_vec)
/// re-sorts the items by the order they were seen.
///
/// # Example
///
/// ```rust
/// # use stepwise::samplers::SampleDeque;
/// // We are interested in the first 20 and last 20 items,
/// // plus 960 items drawn randomly from the middle of the collection.
/// let mut deque = SampleDeque::with_sizes(20, 1000, 20);
///
/// let data_items = (1..=100_000).map(|x| f64::from(x).sqrt()).collect::<Vec<_>>();
/// for (i, data) in data_items.into_iter().enumerate() {
///    // observe the index and the data item
///     deque.sample((i, data));
/// }
///
/// let data_for_plotting: Vec<(usize, f64)> = deque.into_unordered_iter().collect();
/// ```
///
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SampleDeque<T> {
    front: Vec<(usize, T)>,
    middle: SampleVec<(usize, T)>,
    back: VecDeque<(usize, T)>,
    middle_insert_index: usize,
    front_size: usize,
    back_size: usize,
    middle_cap: usize,
    seen: usize,
}

impl<T> Sampler<T> for SampleDeque<T> {
    fn sample(&mut self, item: T) -> SamplingOutcome<T> {
        SampleDeque::sample(self, item)
    }
}

impl<T> SampleDeque<T> {
    /// Creates a new `SampleDeque` with unbounded capacity.
    /// Unbounded sampling means that every item is collected, and the
    /// order of the sample will always match the order of sampling.
    ///
    pub fn unbounded() -> Self {
        Self::with_sizes(usize::MAX, 0, 0)
    }

    /// Creates a new `SampleDeque` with specified *maximum* capacities for all regions.
    /// If usize::MAX appears in any (or all) capacities, the sample is effectively unbounded,
    /// and the collection will as if [`Self::unbounded`] was called.
    /// Which region is unbounded is irrelevant, the behavior is the same.
    /// Any or all of these capacities can be zero.
    pub fn with_sizes(mut front: usize, mut total: usize, mut back: usize) -> Self {
        if front == usize::MAX || total == usize::MAX || back == usize::MAX {
            (front, total, back) = (usize::MAX, 0, 0);
        }
        let middle = total.saturating_sub(front).saturating_sub(back);
        Self {
            front_size: front,
            back_size: back,
            middle_cap: middle,
            front: Vec::new(),
            middle: SampleVec::with_size(middle, RANDOM_SEED),
            back: VecDeque::new(),
            middle_insert_index: 0,
            seen: 0,
        }
    }

    /// Adds an item to the vector, placing it in the appropriate region.
    ///
    /// # Behavior
    /// - If the front region is not full, the item is added to the front.
    /// - If the front is full, the item is added to the back if there is space.
    /// - If both the front and back are full, the oldest item in the back is moved to the middle,
    ///   and the new item is added to the back.
    /// - If the middle region is full, items in the middle are replaced in a round-robin fashion.
    ///
    /// # Arguments
    /// - `item`: The item to be added.
    ///
    /// Performance is O(1), as the middle region is not ordered.
    pub fn sample(&mut self, item: T) -> SamplingOutcome<T> {
        self.seen += 1;
        if self.front.len() < self.front_size {
            self.front.push((self.seen, item));
            return SamplingOutcome::Selected;
        }

        // if front is full, try back
        if self.back.len() < self.back_size {
            self.back.push_back((self.seen, item));
            return SamplingOutcome::Selected;
        }

        // front and back full

        // if tail is being used if theres something to pop,
        // so put in tail and pop oldest to place in middle
        if let Some(old) = self.back.pop_front() {
            self.back.push_back((self.seen, item));
            match self.middle.sample(old) {
                SamplingOutcome::Selected => SamplingOutcome::Selected,
                SamplingOutcome::Replaced(old_middle) => SamplingOutcome::Replaced(old_middle.1),
                SamplingOutcome::Rejected(old_back) => SamplingOutcome::Replaced(old_back.1),
            }
        } else {
            // if back.pop_front fails, yet back was at capacity then back size==0
            match self.middle.sample((self.seen, item)) {
                SamplingOutcome::Selected => SamplingOutcome::Selected,
                SamplingOutcome::Replaced(old_middle) => SamplingOutcome::Replaced(old_middle.1),
                SamplingOutcome::Rejected(item) => SamplingOutcome::Rejected(item.1),
            }
        }
    }

    pub fn sizes(&self) -> (usize, usize, usize) {
        let total = self.middle_cap + self.front_size + self.back_size;
        (self.front_size, total, self.back_size)
    }

    /// Returns the total number of items seen, which will likely exceed the size if
    /// items have been rejected or replaced.
    pub fn seen(&self) -> usize {
        self.seen
    }

    /// Returns the total number of elements in the vector.
    pub fn len(&self) -> usize {
        self.front.len() + self.middle.len() + self.back.len()
    }

    /// Returns `true` if the vector is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a reference iterator over all elements in the vector, starting with the front (in insertion order),
    /// followed by the middle (unordered), and then the back (in insertion order).
    pub fn unordered_iter(&self) -> impl Iterator<Item = &T> {
        self.front
            .iter()
            .map(|item| &item.1)
            .chain(self.middle.as_unordered_slice().iter().map(|item| &item.1))
            .chain(self.back.iter().map(|item| &item.1))
    }

    /// Returns an iterator over all elements in the vector, starting with the front (in insertion order),
    /// followed by the middle (unordered), and then the back (in insertion order).
    pub fn into_unordered_iter(self) -> impl Iterator<Item = T> {
        self.front
            .into_iter()
            .map(|item| item.1)
            .chain(
                self.middle
                    .into_unordered_vec()
                    .into_iter()
                    .map(|item| item.1),
            )
            .chain(self.back.into_iter().map(|item| item.1)) // deque.into_iter is ordered
    }

    /// This is irreversible and consumes the sample. The ordering is by order of sampling.
    pub fn into_ordered_vec(self) -> Vec<T> {
        let mut vec: Vec<_> = self.front.into_iter().map(|(_, t)| t).collect();
        let middle: Vec<_> = self
            .middle
            .into_ordered_vec_by_key(|item| item.0)
            .into_iter()
            .map(|(_, t)| t)
            .collect();
        let back: Vec<_> = self.back.into_iter().map(|(_, t)| t).collect();
        vec.extend(middle);
        vec.extend(back);
        vec
    }

    pub fn get_front(&self, index: usize) -> Option<&T> {
        if index >= self.front.len() {
            return None;
        }
        self.front.get(index).map(|item| &item.1)
    }

    pub fn get_back(&self, index: usize) -> Option<&T> {
        match index {
            i if i >= self.back_size => None,
            i if i >= self.front.len() + self.back.len() => None,
            i if i < self.back.len() => self.back.get(self.back.len() - 1 - i),
            i => self.front.get(self.front.len() - 1 - (i - self.back.len())),
        }
        .map(|item| &item.1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_deque_no_middle() {
        let mut deque = SampleDeque::with_sizes(3, 5, 2);
        assert_eq!(deque.sizes(), (3, 5, 2));

        for i in 1..=5 {
            // first 5 are accepted into front and back
            assert_eq!(deque.sample(i), SamplingOutcome::Selected);
        }
        for i in 6..=10 {
            // only back capacity of 2, so each i replaces an i-2
            assert_eq!(deque.sample(i), SamplingOutcome::Replaced(i - 2));
        }

        assert_eq!(deque.len(), 5);
        assert_eq!(deque.front.len(), 3);
        assert_eq!(deque.middle.len(), 0);
        assert_eq!(deque.back.len(), 2);

        let front: Vec<_> = deque.front.to_vec();
        let back: Vec<_> = deque.back.iter().copied().collect();

        assert_eq!(front, vec![(1, 1), (2, 2), (3, 3)]);
        assert_eq!(back, vec![(9, 9), (10, 10)]);
    }

    #[test]
    fn sample_deque_with_middle() {
        let mut deque = SampleDeque::with_sizes(3, 10, 2);
        assert_eq!(deque.sizes(), (3, 10, 2));
        assert!(deque.get_front(0).is_none());
        assert!(deque.get_back(0).is_none());

        assert_eq!(deque.sample(1), SamplingOutcome::Selected);
        assert_eq!(deque.get_front(0), Some(&1));
        assert_eq!(deque.get_front(1), None);
        assert_eq!(deque.get_back(0), Some(&1));
        assert_eq!(deque.get_back(1), None);

        assert_eq!(deque.sample(2), SamplingOutcome::Selected);
        assert_eq!(deque.get_front(0), Some(&1));
        assert_eq!(deque.get_front(1), Some(&2));
        assert_eq!(deque.get_front(3), None);
        assert_eq!(deque.get_back(0), Some(&2));
        assert_eq!(deque.get_back(1), Some(&1));
        assert_eq!(deque.get_back(2), None);

        for i in 3..=5 {
            // first 5 are accepted into front and back
            assert_eq!(deque.sample(i), SamplingOutcome::Selected);
        }
        assert_eq!(deque.len(), 5);
        assert_eq!(deque.front.len(), 3);
        assert_eq!(deque.middle.len(), 0);
        assert_eq!(deque.back.len(), 2);

        assert_eq!(deque.get_front(0), Some(&1));
        assert_eq!(deque.get_front(1), Some(&2));
        assert_eq!(deque.get_front(2), Some(&3));
        assert_eq!(deque.get_front(3), None);

        assert_eq!(deque.get_back(0), Some(&5));
        assert_eq!(deque.get_back(1), Some(&4));
        assert_eq!(deque.get_back(2), None);

        for i in 6..=10 {
            assert_eq!(deque.sample(i), SamplingOutcome::Selected);
        }

        assert_eq!(deque.len(), 10);
        assert_eq!(deque.front.len(), 3);
        assert_eq!(deque.middle.len(), 5);
        assert_eq!(deque.back.len(), 2);

        // check front and back
        let front: Vec<_> = deque.front.to_vec();
        let back: Vec<_> = deque.back.iter().copied().collect();
        assert_eq!(front, vec![(1, 1), (2, 2), (3, 3)]);
        assert_eq!(back, vec![(9, 9), (10, 10)]);

        // check middle
        let mut middle: Vec<_> = deque.middle.as_unordered_slice().to_vec();
        middle.sort();
        assert_eq!(middle, vec![(4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]);

        let mut evicted = vec![];

        for i in 11..=16 {
            let outcome = deque.sample(i);
            match outcome {
                SamplingOutcome::Selected => panic!("expected Replaced or Rejected"),
                SamplingOutcome::Replaced(a) => {
                    assert!((10..=i).contains(&a) || (4..=8).contains(&a));
                    evicted.push(a);
                }
                SamplingOutcome::Rejected(rej) => assert!(rej == i),
            }
        }

        // check front and back
        let front: Vec<_> = deque.front.to_vec();
        let back: Vec<_> = deque.back.iter().copied().collect();
        assert_eq!(front, vec![(1, 1), (2, 2), (3, 3)]);
        assert_eq!(back, vec![(15, 15), (16, 16)]);

        // check middle/evicted
        assert_eq!(middle.len(), 5);
        let mut middle: Vec<_> = deque.middle.as_unordered_slice().to_vec();
        middle.sort();
        evicted.sort();
        // assert!(evicted.all(|e| , ));
    }

    #[test]
    fn sample_deque_unbounded() {
        let mut deque = SampleDeque::unbounded();
        for i in 1..=10 {
            assert_eq!(deque.sample(i), SamplingOutcome::Selected);
        }

        assert_eq!(deque.len(), 10);
        assert_eq!(deque.front.len(), 10, "{deque:#?}");
        assert_eq!(deque.middle.len(), 0);
        assert_eq!(deque.back.len(), 0);

        let head: Vec<_> = deque.front.to_vec();
        assert_eq!(
            head,
            vec![
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (6, 6),
                (7, 7),
                (8, 8),
                (9, 9),
                (10, 10)
            ]
        );
    }
}
