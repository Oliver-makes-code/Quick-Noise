use std::sync::LazyLock;
use std::simd::StdFloat;
use std::time::Instant;
use crate::simd::arch_simd::{ArchSimd, SimdInfo};
use crate::simd::simd_array::SimdArray;
use crate::math::vec::{Vec2, Vec3};
use crate::math::random::Random;
use image;

macro_rules! compute_lerps_2d {
    ($d_vecs:expr, $interpolations:expr, $x_upper_increment:expr, $x_lower_increment:expr, 
     $x_weighted_increment_vec:expr, $weight_vec:expr, $y_it:expr) => {{
        let y_lerp = $interpolations.y.load_simd($y_it);
        let x_tl = $d_vecs.tl().x.load_simd($y_it);
        let x_tr = $d_vecs.tr().x.load_simd($y_it);
        let x_bl = $d_vecs.bl().x.load_simd($y_it);
        let x_br = $d_vecs.br().x.load_simd($y_it);
        let y_tl = $d_vecs.tl().y.load_simd($y_it);
        let y_tr = $d_vecs.tr().y.load_simd($y_it);
        let y_bl = $d_vecs.bl().y.load_simd($y_it);
        let y_br = $d_vecs.br().y.load_simd($y_it);

        let prod_sum_tl = x_tl.mul_add($x_upper_increment, y_tl);
        let prod_sum_tr = x_tr.mul_add($x_upper_increment, y_tr);
        let prod_sum_bl = x_bl.mul_add($x_lower_increment, y_bl);
        let prod_sum_br = x_br.mul_add($x_lower_increment, y_br);

        let base_lerp_top = y_lerp.mul_add(prod_sum_tr - prod_sum_tl, prod_sum_tl) * $weight_vec;
        let base_lerp_bottom = y_lerp.mul_add(prod_sum_br - prod_sum_bl, prod_sum_bl) * $weight_vec;
        let base_lerp_dif = base_lerp_bottom - base_lerp_top;

        let x_offset_tl = x_tl * $x_weighted_increment_vec;
        let x_offset_tr = x_tr * $x_weighted_increment_vec;
        let x_offset_bl = x_bl * $x_weighted_increment_vec;
        let x_offset_br = x_br * $x_weighted_increment_vec;

        let x_offset_lerp_top = y_lerp.mul_add(x_offset_tr - x_offset_tl, x_offset_tl);
        let x_offset_lerp_bottom = y_lerp.mul_add(x_offset_br - x_offset_bl, x_offset_bl);
        let x_offset_lerp_dif = x_offset_lerp_bottom - x_offset_lerp_top;

        (base_lerp_top, base_lerp_dif, x_offset_lerp_top, x_offset_lerp_dif)
    }};
}

macro_rules! compute_results_2d {
    ($result:expr, $x_lerp:expr, $base_lerp_top:expr, $base_lerp_dif:expr, 
     $x_offset_lerp_top:expr, $x_offset_lerp_dif:expr, $index:expr, $initialize:expr) => {{
        let output = $x_lerp.mul_add($base_lerp_dif, $base_lerp_top);
        let val = if $initialize { output } else { output + $result.load_simd($index) };
        $result.store_simd($index, val);
        $base_lerp_dif += $x_offset_lerp_dif;
        $base_lerp_top += $x_offset_lerp_top;
    }};
}

const ROW_SIZE: usize = 32;
const MAP_SIZE: usize = 1024;
const VOL_SIZE: usize = 32768;
const NUM_DIRECTIONS: usize = 16;
const LO_EPSILON: f64 = 1e-4;
const HI_EPSILON: f64 = 1.0 - 1e-4;

const GRADIENTS_3D_LOOKUP: [(f64, f64, f64); 12] = [
    (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, -1.0, 0.0),
    (1.0, 0.0, 1.0), (-1.0, 0.0, 1.0), (1.0, 0.0, -1.0), (-1.0, 0.0, -1.0),
    (0.0, 1.0, 1.0), (0.0, -1.0, 1.0), (0.0, 1.0, -1.0), (0.0, -1.0, -1.0)
];

type PerlinVec = SimdArray<f32, ROW_SIZE>;
type PerlinMap = SimdArray<f32, MAP_SIZE>;
type PerlinVol = SimdArray<f32, VOL_SIZE>;

type PerlinVecPair = Vec2<PerlinVec>;
type PerlinVecTriple = Vec3<PerlinVec>;

// static GRADIENTS_2D: LazyLock<[Vec2<f64>; NUM_DIRECTIONS]> = LazyLock::new(|| {
//     let mut gradients: [Vec2<f64>; NUM_DIRECTIONS] = [Vec2::<f64> {x: 0.0, y: 0.0}; NUM_DIRECTIONS];

//     let scale: f64 = 2.0_f64.sqrt();
//     for i in 0..NUM_DIRECTIONS {
//         let angle: f64 = (2.0 * PI) * (i as f64 / NUM_DIRECTIONS as f64);
//         let x: f64 = angle.cos() * scale;
//         let y: f64 = angle.sin() * scale;
//         gradients[i] = Vec2::<f64> {x, y};
//     }

//     gradients
// });

const GRADIENTS_2D: [Vec2<f32>; 16] = [
    Vec2 { x:  1.4142135623730951, y:  0.0000000000000000 },
    Vec2 { x:  1.3065629648763766, y:  0.5411961001338174 },
    Vec2 { x:  1.0000000000000002, y:  1.0000000000000002 },
    Vec2 { x:  0.5411961001338176, y:  1.3065629648763766 },
    Vec2 { x:  0.0000000000000001, y:  1.4142135623730951 },
    Vec2 { x: -0.5411961001338172, y:  1.3065629648763768 },
    Vec2 { x: -0.9999999999999998, y:  1.0000000000000004 },
    Vec2 { x: -1.3065629648763764, y:  0.5411961001338180 },
    Vec2 { x: -1.4142135623730951, y:  0.0000000000000000 },
    Vec2 { x: -1.3065629648763768, y: -0.5411961001338169 },
    Vec2 { x: -1.0000000000000004, y: -0.9999999999999998 },
    Vec2 { x: -0.5411961001338183, y: -1.3065629648763764 },
    Vec2 { x: -0.0000000000000002, y: -1.4142135623730951 },
    Vec2 { x:  0.5411961001338166, y: -1.3065629648763771 },
    Vec2 { x:  0.9999999999999996, y: -1.0000000000000007 },
    Vec2 { x:  1.3065629648763771, y: -0.5411961001338161 },
];

static GRADIENTS_3D: LazyLock<[Vec3<f64>; NUM_DIRECTIONS]> = LazyLock::new(|| {
    let mut gradients: [Vec3<f64>; NUM_DIRECTIONS] = [Vec3::<f64>::new(0.0, 0.0, 0.0); NUM_DIRECTIONS];

    for i in (0..NUM_DIRECTIONS).step_by(12) {
        for j in 0..12 {
            gradients[i] = GRADIENTS_3D_LOOKUP[j].into();
        }
    }

    gradients
});

#[derive(Copy, Clone)]
pub struct Octave2D {
    scale: Vec2<f32>,
    weight: f32,
}

impl Octave2D {
    pub fn new(scale: Vec2<f32>, weight: f32) -> Self {
        Self { scale, weight }
    }

    pub fn splat(scale: f32, weight: f32) -> Self {
        Self { scale: Vec2::<f32>::new(scale, scale), weight }
    }
}

impl From<(f32, f32)> for Octave2D {
    fn from((scale, weight): (f32, f32)) -> Self {
        Octave2D::new((scale, scale).into(), weight)
    }
}

impl From<((f32, f32), f32)> for Octave2D {
    fn from(((x_scale, y_scale), weight): ((f32, f32), f32)) -> Self {
        Octave2D::new((x_scale, y_scale).into(), weight)
    }
}

impl From<&Octave2D> for Octave2D {
    fn from(octave: &Octave2D) -> Self {
        octave.clone()
    }
}

#[derive(Copy, Clone)]
pub struct Octave3D {
    scale: Vec3<f32>,
    weight: f32,
}

impl Octave3D {
    pub fn new(scale: Vec3<f32>, weight: f32) -> Self {
        Self { scale, weight }
    }

    pub fn splat(scale: f32, weight: f32) -> Self {
        Self { scale: Vec3::<f32>::new(scale, scale, scale), weight }
    }
}

struct PerlinContainer2D {
    vecs: [PerlinVecPair; 4],
    tl: usize,
    tr: usize,
    bl: usize,
    br: usize,
}

impl PerlinContainer2D {
    pub fn new_uninit() -> Self {
        PerlinContainer2D {
            vecs: [
                PerlinVecPair::new(PerlinVec::new_uninit(), PerlinVec::new_uninit()),
                PerlinVecPair::new(PerlinVec::new_uninit(), PerlinVec::new_uninit()),
                PerlinVecPair::new(PerlinVec::new_uninit(), PerlinVec::new_uninit()),
                PerlinVecPair::new(PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            ],
            tl: 0,
            tr: 1,
            bl: 2,
            br: 3,
        }
    }

    pub fn tl(&self) -> &PerlinVecPair { unsafe { &self.vecs.get_unchecked(self.tl) } }
    pub fn tr(&self) -> &PerlinVecPair { unsafe { &self.vecs.get_unchecked(self.tr) } }
    pub fn bl(&self) -> &PerlinVecPair { unsafe { &self.vecs.get_unchecked(self.bl) } }
    pub fn br(&self) -> &PerlinVecPair { unsafe { &self.vecs.get_unchecked(self.br) } }

    pub fn tl_tr_mut(&mut self) -> (&mut PerlinVecPair, &mut PerlinVecPair) {
        debug_assert!(self.tl < self.tr);
        debug_assert!(self.tr < self.vecs.len());
        unsafe {
            let ptr = self.vecs.as_mut_ptr();
            (&mut *ptr.add(self.tl), &mut *ptr.add(self.tr))
        }
    }

    pub fn bl_br_mut(&mut self) -> (&mut PerlinVecPair, &mut PerlinVecPair) {
        debug_assert!(self.bl < self.br);
        debug_assert!(self.br < self.vecs.len());
        unsafe {
            let ptr = self.vecs.as_mut_ptr();
            (&mut *ptr.add(self.bl), &mut *ptr.add(self.br))
        }
    }

    pub fn swap_top_bottom(&mut self) {
        std::mem::swap(&mut self.tl, &mut self.bl);
        std::mem::swap(&mut self.tr, &mut self.br);
    }
}

struct PerlinContainer3D {
    tlf: PerlinVecTriple, // Top left front.
    trf: PerlinVecTriple, // Top right front.
    blf: PerlinVecTriple, // Bottom left front.
    brf: PerlinVecTriple, // Bottom right front.
    tlb: PerlinVecTriple, // Top left back.
    trb: PerlinVecTriple, // Top right back.
    blb: PerlinVecTriple, // Bottom left back.
    brb: PerlinVecTriple, // Bottom right back.
}

impl PerlinContainer3D {
    pub fn new_uninit() -> Self {
        PerlinContainer3D {
            tlf: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            trf: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            blf: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            brf: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            tlb: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            trb: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            blb: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
            brb: PerlinVecTriple::new(PerlinVec::new_uninit(), PerlinVec::new_uninit(), PerlinVec::new_uninit()),
        }
    }
}

pub struct Perlin {
    random_gen: Random
}

impl Perlin {

    pub fn new(seed: i64) -> Self {
        Self {
            random_gen: Random::new(seed as u64),
        }
    }

    pub fn write_height_map(
        &mut self,
        path: &str,
        dimension: usize,
        octaves: u32,
        scale: f32,
        lacunarity: f32,
        persistence: f32
    ) {
        let mut pixels = Vec::<u8>::new();
        pixels.resize(dimension * dimension * MAP_SIZE, 0);
        
        for x in 0..dimension {
            let x_offset = x * dimension * MAP_SIZE;
            for y in 0..dimension {
                let y_offset = y * ROW_SIZE;
                let mut noise = self.noise_2d((x as i32, y as i32).into(), octaves, scale, 1.0, lacunarity, persistence, 1, 0.0);

                noise = (noise + PerlinMap::new(1.0)) * PerlinMap::new(127.5);

                for dx in 0..ROW_SIZE {
                    let offset = x_offset + y_offset + dx * ROW_SIZE * dimension;
                    for dy in 0..ROW_SIZE {
                        pixels[offset + dy] = noise[dx * ROW_SIZE + dy] as u8;
                    }
                }
            }
        }

        let pixel_dimension = (dimension * ROW_SIZE) as u32;
        image::save_buffer(
            &path,
            &pixels,
            pixel_dimension,
            pixel_dimension,
            image::ColorType::L8
        ).expect("Failed to write height map!");

        println!("Wrote height map to {path}!");
    }

    pub fn write_height_map_octaves(
        &mut self,
        path: &str,
        dimension: usize,
        octaves: impl IntoIterator<Item = impl Into<Octave2D>>,
        channel: i32,
    ) {
        let octaves_vec: Vec<Octave2D> = octaves.into_iter().map(Into::into).collect();
        let mut pixels = Vec::<u8>::new();
        pixels.resize(dimension * dimension * MAP_SIZE, 0);
        
        for x in 0..dimension {
            let x_offset = x * dimension * MAP_SIZE;
            for y in 0..dimension {
                let y_offset = y * ROW_SIZE;
                let mut noise = self.noise_2d_octaves((x as i32, y as i32).into(), &octaves_vec, 1.0, channel, 0.0);

                noise = (noise + PerlinMap::new(1.0)) * PerlinMap::new(127.5);

                for dx in 0..ROW_SIZE {
                    let offset = x_offset + y_offset + dx * ROW_SIZE * dimension;
                    for dy in 0..ROW_SIZE {
                        pixels[offset + dy] = noise[dx * ROW_SIZE + dy] as u8;
                    }
                }
            }
        }

        let pixel_dimension = (dimension * ROW_SIZE) as u32;
        image::save_buffer(
            &path,
            &pixels,
            pixel_dimension,
            pixel_dimension,
            image::ColorType::L8
        ).expect("Failed to write height map!");

        println!("Wrote height map to {path}!");
    }

    pub fn profile_noise_2d() {
        const NUM_LOOPS: usize = 10000000;
        const SAMPLE_SIZE: usize = 1024;
        let mut code: f32 = 0.0;

        let mut perlin = Perlin::new(0);

        let start = Instant::now();
        for i in 0..NUM_LOOPS {
            code += perlin.noise_2d((i as i32, i as i32).into(), 1, 32.0, 1.0, 2.0, 0.5, 1, 0.0)[i & 0xFF];
        }
        let elapsed = start.elapsed();
        let ms_elapsed = elapsed.as_millis();
        let total = NUM_LOOPS * SAMPLE_SIZE;

        let elapsed_per_loop = elapsed.as_nanos() as u64 / NUM_LOOPS as u64;

        let samples_per_second = (total as f64 / elapsed.as_secs_f64()) as u64;

        println!(
            "-+- Completed 2D Perlin Noise Bench -+-\n\
            -> {total} samples in {ms_elapsed} ms!\n\
            -> {elapsed_per_loop} ns per {SAMPLE_SIZE} samples!\n\
            -> {samples_per_second} samples per second!\n\
            Code: {code}"
        );

    }

    pub fn noise_2d(
        &mut self,
        pos: Vec2<i32>,
        octaves: u32,
        scale: f32,
        amplitude: f32,
        lacunarity: f32,
        persistence: f32,
        channel: i32,
        octave_offset: f32,
    ) -> PerlinMap {
        let channel_seed: u64 = Random::static_mix_u64(channel as u64);
        let mut result: PerlinMap = PerlinMap::new_uninit();

        let mut weight_sum = amplitude;
        let mut cur_weight = amplitude;
        for _ in 1..octaves {
            cur_weight *= persistence;
            weight_sum += cur_weight;
        }
        let weight_coef = 1.0 / weight_sum;

        let mut cur_octave = Octave2D::splat(scale, 1.0);
        
        self.single_octave_2d::<true>(&mut result, pos, &cur_octave, weight_coef, channel_seed, octave_offset);

        for _ in 1..octaves {
            cur_octave.scale /= lacunarity;
            cur_octave.weight *= persistence;

            self.single_octave_2d::<false>(&mut result, pos, &cur_octave, weight_coef, channel_seed, octave_offset);
        }

        result
    }

    pub fn noise_2d_octaves(
        &mut self,
        pos: Vec2<i32>,
        octaves: impl IntoIterator<Item = impl Into<Octave2D>>,
        amplitude: f32,
        channel: i32,
        octave_offset: f32,
    ) -> PerlinMap {
        let octaves_vec: Vec<Octave2D> = octaves.into_iter().map(Into::into).collect();
        let channel_seed: u64 = Random::static_mix_u64(channel as u64);
        let mut result: PerlinMap = PerlinMap::new_uninit();

        let mut weight_sum = 0.0;
        for octave in &octaves_vec {
            weight_sum += octave.weight;
        }
        let weight_coef = amplitude / weight_sum;

        self.single_octave_2d::<true>(&mut result, pos, &octaves_vec[0], weight_coef, channel_seed, octave_offset);
        for i in 1..octaves_vec.len() {
            self.single_octave_2d::<false>(&mut result, pos, &octaves_vec[i], weight_coef, channel_seed, octave_offset);
        }

        result
    }
    
    fn single_octave_2d<const INITIALIZE: bool>(
        &mut self,
        result: &mut PerlinMap,
        pos: Vec2<i32>,
        octave: &Octave2D,
        weight_coef: f32,
        channel_seed: u64,
        octave_offset: f32,
    ) {
        let increment: Vec2<f32> = 1.0 / octave.scale;
        let block_pos: Vec2<i32> = pos * 32;
        let weight: f32 = octave.weight * weight_coef;

        let grid_start: Vec2<i32> = ((block_pos + 1).as_f32() * increment + LO_EPSILON as f32).floor().as_i32();
        let frac_start: Vec2<f32> = Vec2::max_float(block_pos.as_f32() * increment - grid_start.as_f32(), Vec2::splat(0.0));

        let distances: PerlinVecPair = PerlinVecPair {
            x: PerlinVec::iota_custom(frac_start.x + LO_EPSILON as f32, increment.x).fract(),
            y: PerlinVec::iota_custom(frac_start.y + LO_EPSILON as f32, increment.y).fract(),
        };

        let interpolations: PerlinVecPair = PerlinVecPair {
            x: distances.x.quintic_lerp(),
            y: distances.y.quintic_lerp(),
        };

        // Note: Octave offset does not currently work.
        self.random_gen.set_channel(channel_seed ^ (octave.scale + octave_offset).sum() as u64);

        let num_loops: Vec2<u32> = (frac_start + increment * ROW_SIZE as f32 - LO_EPSILON as f32).ceil().as_u32();
        let next_index_offset: Vec2<f32> = (1.0 - frac_start) * octave.scale + HI_EPSILON as f32;

        let mut d_vecs: PerlinContainer2D = PerlinContainer2D::new_uninit();

        let (tl, tr) = d_vecs.tl_tr_mut();
        self.set_gridpoints_2d(tl, tr, grid_start.x, grid_start.y, next_index_offset.y, octave.scale.y, num_loops.y, &distances.y);

        let mut x_cur_index: usize = 0;
        for x_it in 0..num_loops.x {
            let x_next_index = ((x_it as f32 * octave.scale.x + next_index_offset.x) as u32).min(ROW_SIZE as u32) as usize;

            let x_cur_frac_start = unsafe { distances.x.get_unchecked(x_cur_index) };
            let x_chunk_size = x_next_index - x_cur_index;

            let (bl, br) = d_vecs.bl_br_mut();
            self.set_gridpoints_2d(bl, br, grid_start.x + x_it as i32 + 1, grid_start.y, next_index_offset.y, octave.scale.y, num_loops.y, &distances.y);
        
            Self::compute_noise_from_vecs_2d::<INITIALIZE>(
                &d_vecs, x_cur_frac_start, increment.x,
                x_chunk_size, &interpolations, x_cur_index, weight, result
            );

            d_vecs.swap_top_bottom();

            if x_next_index == 32 { break; }
            x_cur_index = x_next_index;
        }
    }

    #[inline(never)]
    fn set_gridpoints_2d (
        &mut self,
        left: &mut PerlinVecPair,
        right: &mut PerlinVecPair,
        x_start: i32,
        y_start: i32,
        y_next_index_offset: f32,
        y_scale: f32,
        y_num_loops: u32,
        y_distances: &PerlinVec,
    ) {
        let mut left_grad = (self.random_gen.mix_i32_pair(x_start, y_start) % NUM_DIRECTIONS as u64) as usize;
        let mut cur_index: usize = 0;

        let mut arrays = [
            &mut left.x, &mut left.y,
            &mut right.x, &mut right.y,
        ];

        for y_it in 0..y_num_loops {
            let next_index = ((y_it as f32 * y_scale + y_next_index_offset) as usize).min(ROW_SIZE as usize);
            let set_amount = next_index - cur_index;

            let right_grad = (self.random_gen.mix_i32_pair(x_start, y_start + y_it as i32 + 1) % NUM_DIRECTIONS as u64) as usize;

            let values = [
                GRADIENTS_2D[left_grad].x, GRADIENTS_2D[left_grad].y,
                GRADIENTS_2D[right_grad].x, GRADIENTS_2D[right_grad].y,
            ];

            PerlinVec::multiset_many::<4>(&mut arrays, &values, cur_index, set_amount as isize);

            left_grad = right_grad;
            cur_index = next_index;
        }

        left.y *= *y_distances;
        right.y *= *y_distances - PerlinVec::new(1.0);
    }

    #[inline(never)]
    fn compute_noise_from_vecs_2d<const INITIALIZE: bool>(
        d_vecs: &PerlinContainer2D,
        x_frac_start: f32,
        x_increment: f32,
        x_chunk_size: usize,
        interpolations: &PerlinVecPair,
        x_start_index: usize,
        weight: f32,
        result: &mut PerlinMap,
    ) {
        let weight_vec = ArchSimd::splat(weight);
        let x_weighted_increment_vec = ArchSimd::splat(x_increment) * weight_vec;
        let x_upper_increment = ArchSimd::splat(x_frac_start);
        let x_lower_increment = ArchSimd::splat(x_frac_start - 1.0);

        #[cfg(target_feature = "neon")]
        {
            for y_it in (0..ROW_SIZE).step_by(f32::LANES * 4) {
                let (mut base_lerp_top_1, mut base_lerp_dif_1, x_offset_lerp_top_1, x_offset_lerp_dif_1) =
                    compute_lerps_2d!(d_vecs, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it);
                let (mut base_lerp_top_2, mut base_lerp_dif_2, x_offset_lerp_top_2, x_offset_lerp_dif_2) =
                    compute_lerps_2d!(d_vecs, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it + f32::LANES);
                let (mut base_lerp_top_3, mut base_lerp_dif_3, x_offset_lerp_top_3, x_offset_lerp_dif_3) =
                    compute_lerps_2d!(d_vecs, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it + f32::LANES * 2);
                let (mut base_lerp_top_4, mut base_lerp_dif_4, x_offset_lerp_top_4, x_offset_lerp_dif_4) =
                    compute_lerps_2d!(d_vecs, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it + f32::LANES * 3);
    
                for x_it in x_start_index..x_start_index + x_chunk_size {
                    let x_lerp = ArchSimd::splat(unsafe { interpolations.x.get_unchecked(x_it) } );
                    compute_results_2d!(result, x_lerp, base_lerp_top_1, base_lerp_dif_1, x_offset_lerp_top_1, x_offset_lerp_dif_1, x_it * ROW_SIZE + y_it, INITIALIZE);
                    compute_results_2d!(result, x_lerp, base_lerp_top_2, base_lerp_dif_2, x_offset_lerp_top_2, x_offset_lerp_dif_2, x_it * ROW_SIZE + y_it + f32::LANES, INITIALIZE);
                    compute_results_2d!(result, x_lerp, base_lerp_top_3, base_lerp_dif_3, x_offset_lerp_top_3, x_offset_lerp_dif_3, x_it * ROW_SIZE + y_it + f32::LANES * 2, INITIALIZE);
                    compute_results_2d!(result, x_lerp, base_lerp_top_4, base_lerp_dif_4, x_offset_lerp_top_4, x_offset_lerp_dif_4, x_it * ROW_SIZE + y_it + f32::LANES * 3, INITIALIZE);
                }
            }
        }

        #[cfg(not(target_feature = "neon"))]
        {
            for y_it in (0..ROW_SIZE).step_by(f32::LANES * 2) {
                let (mut base_lerp_top_1, mut base_lerp_dif_1, x_offset_lerp_top_1, x_offset_lerp_dif_1) =
                    compute_lerps_2d!(d_vecs, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it);
                let (mut base_lerp_top_2, mut base_lerp_dif_2, x_offset_lerp_top_2, x_offset_lerp_dif_2) =
                    compute_lerps_2d!(d_vecs, interpolations, x_upper_increment, x_lower_increment, x_weighted_increment_vec, weight_vec, y_it + f32::LANES);
    
                for x_it in x_start_index..x_start_index + x_chunk_size {
                    let x_lerp = ArchSimd::splat(unsafe { interpolations.x.get_unchecked(x_it) } );
                    compute_results_2d!(result, x_lerp, base_lerp_top_1, base_lerp_dif_1, x_offset_lerp_top_1, x_offset_lerp_dif_1, x_it * ROW_SIZE + y_it, INITIALIZE);
                    compute_results_2d!(result, x_lerp, base_lerp_top_2, base_lerp_dif_2, x_offset_lerp_top_2, x_offset_lerp_dif_2, x_it * ROW_SIZE + y_it + f32::LANES, INITIALIZE);
                }
            }
        }
    }
}
