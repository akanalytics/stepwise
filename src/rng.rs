#[derive(Debug, Clone, PartialEq, Eq)]

/// Implmentation of PCG-XSH-RR 64/32
/// <https://en.wikipedia.org/wiki/Permuted_congruential_generator>
pub(crate) struct TinyRng {
    state: u64,
}

impl TinyRng {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);

        // output function: xorshift and rotate
        let xorshifted = (((self.state >> 18) ^ self.state) >> 27) as u32;
        let rot = (self.state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Generates a random u64 by combining two u32 values.
    pub fn next_u64(&mut self) -> u64 {
        let x = u64::from(self.next_u32());
        let y = u64::from(self.next_u32());
        (y << 32) | x
    }

    /// Generates a random u64 in the range [0, n) using multiplication and bit-shift scaling.
    pub fn gen_range(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assert_approx_eq;

    #[test]
    fn test_tiny_rng() {
        let mut rng = TinyRng::new(42);
        let mut rng2 = TinyRng::new(42);
        for _ in 0..1000 {
            assert_eq!(rng.next_u32(), rng2.next_u32());
            assert_eq!(rng.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_gen_range() {
        let mut rng = TinyRng::new(42);
        let mut counts = vec![0; 10];
        for _ in 0..10000 {
            counts[rng.gen_range(10)] += 1;
        }
        for count in counts {
            println!("{count}");
            // Check that the count is within 10% of the expected value
            assert_approx_eq!(count as f64, 1000.0, 100.0);
        }
    }
}
