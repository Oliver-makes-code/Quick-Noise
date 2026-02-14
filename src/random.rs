// Primitive but fast module for generating random-looking outputs.

pub struct Random {
    core_seed: u64,
    channel_seed: u64
}

impl Random {
    pub fn new(seed: u64) -> Self {
        let init_core_seed: u64 = Self::static_mix_u64(seed);
        let init_channel_seed: u64 = Self::static_mix_u64(init_core_seed);

        Self { 
            core_seed: init_core_seed,
            channel_seed: init_channel_seed
        }
    }

    pub fn set_channel(&mut self) {
        self.channel_seed = Self::static_mix_u64(self.core_seed)
    }

    // --- Raw Mixers ---

    pub fn mix_u64(&self, data: u64) -> u64 {
        Self::mix_u64_impl(data ^ self.channel_seed)
    }

    pub fn mix_u64_pair(&self, data1: u64, data2: u64) -> u64 {
        Self::mix_u64_pair_impl(data1 ^ self.channel_seed, data2)
    }

    pub fn mix_u32(&self, data: u32) -> u32 {
        Self::mix_bits_32_impl(data ^ self.core_seed as u32)
    }

    // --- Mix Variants ---

    pub fn mix_i32_pair(&self, data1: i32, data2: i32) -> u64 {
        self.mix_u64(Self::combine_i32_pair(data1, data2))
    }

    pub fn mix_i32_triple(&self, data1: i32, data2: i32, data3: i32) -> u64 {
        let concat_data: u64 = Self::combine_i32_pair(data1, data2);
        self.mix_u64_pair(concat_data, (data3 as u64) << 32)
    }
    
    // --- Static Mixers ---

    pub fn static_mix_u64(data: u64) -> u64 {
        Self::mix_u64_impl(data ^ 0x9e3779b97f4a7c15)
    }

    pub fn static_mix_u32(data: u32) -> u32 {
        Self::mix_bits_32_impl(data ^ 0x7f4a7c15)
    }

    // --- Private Helpers ---
    
    fn combine_i32_pair(data1: i32, data2: i32) -> u64 {
        (data1 as u64) | ((data2 as u64) << 32)
    }

    // --- Bit mixing implementations | From MurmurHash ---

    fn mix_u64_impl(mut data: u64) -> u64 {
        data ^= data >> 33;
        data = data.wrapping_mul(0xff51afd7ed558ccd);
        data ^= data >> 33;
        data = data.wrapping_mul(0xc4ceb9fe1a85ec53);
        data ^= data >> 33;
        data
    }

    fn mix_u64_pair_impl(mut data1: u64, data2: u64) -> u64 {
        data1 ^= data1 >> 33;
        data1 = data1.wrapping_mul(0xff51afd7ed558ccd ^ data2);
        data1 ^= data1 >> 33;
        data1 = data1.wrapping_mul(0xc4ceb9fe1a85ec53 ^ data2);
        data1 ^= data1 >> 33;
        data1
    }

    fn mix_bits_32_impl(mut data: u32) -> u32 {
        data ^= data >> 16;
        data = data.wrapping_mul(0x85ebca6b);
        data ^= data >> 13;
        data = data.wrapping_mul(0xc2b2ae35);
        data ^= data >> 16;
        data
    }
}
