use std::fmt;
use std::mem::MaybeUninit;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};
use std::simd::prelude::{SimdFloat, SimdPartialOrd};
use std::simd::{Mask, Simd, SimdElement, StdFloat};
use crate::simd::simd_array::fmt::Debug;
use crate::simd::arch_simd::{ArchSimd, SimdInfo};
use num_traits::NumCast;
use num_traits::float::Float;

// === Macro Helpers ===

macro_rules! impl_simd_array_op {
    ($trait:ident, $assign_trait:ident, $method:ident, $assign_method:ident, $op:tt, $assign_op:tt) => {
        impl<T: SimdInfo, const N: usize> $trait for SimdArray<T, N>
        where
            [(); T::LANES]:,
            T: $trait<Output = T>,
            ArchSimd<T>: $trait<Output = ArchSimd<T>>,
        {
            type Output = SimdArray<T, N>;
            fn $method(self, other: SimdArray<T, N>) -> SimdArray<T, N> {
                let mut result = SimdArray::<T, N>::new_uninit();
                for i in (0..Self::TAIL_START).step_by(T::LANES) {
                    let caller_vec = self.load_simd(i);
                    let other_vec = other.load_simd(i);
                    result.store_simd(i, caller_vec $op other_vec);
                }
                if Self::HAS_TAIL {
                    let caller_vec = self.load_simd(Self::TAIL_START);
                    let other_vec = other.load_simd(Self::TAIL_START);
                    result.partial_store_simd(Self::TAIL_START, caller_vec $op other_vec, Self::TAIL_SIZE);
                }
                result
            }
        }

        impl<T: SimdInfo, const N: usize> $assign_trait for SimdArray<T, N>
        where
            [(); T::LANES]:,
            T: $assign_trait,
            ArchSimd<T>: $assign_trait,
        {
            fn $assign_method(&mut self, other: SimdArray<T, N>) {
                for i in (0..Self::TAIL_START).step_by(T::LANES) {
                    let mut caller_vec = self.load_simd(i);
                    caller_vec $assign_op other.load_simd(i);
                    self.store_simd(i, caller_vec);
                }
                if Self::HAS_TAIL {
                    let mut caller_vec = self.load_simd(Self::TAIL_START);
                    caller_vec $assign_op other.load_simd(Self::TAIL_START);
                    self.partial_store_simd(Self::TAIL_START, caller_vec, Self::TAIL_SIZE);
                }
            }
        }
    }
}

#[repr(align(64))]
#[derive(Copy)]
pub struct SimdArray<T: SimdInfo, const N: usize> {
    pub data: [MaybeUninit<T>; N],
}

// === Tail Info ===

pub trait TailInfo {
    const TAIL_SIZE: usize;
    const TAIL_START: usize;
    const HAS_TAIL: bool;
}

impl<T: SimdInfo, const N: usize> TailInfo for SimdArray<T, N> {
    const TAIL_SIZE: usize = N % T::LANES;
    const TAIL_START: usize = N - Self::TAIL_SIZE;
    const HAS_TAIL: bool = Self::TAIL_SIZE > 0;
}

// === Constructors ===

impl<T: SimdInfo, const N: usize> SimdArray<T, N> {
    pub fn new_uninit() -> Self {
        Self {
            data: unsafe { MaybeUninit::uninit().assume_init() },
        }
    }
}

impl<T: SimdInfo + Copy, const N: usize> SimdArray<T, N> {
    pub fn new(value: T) -> Self {
        Self {
            data: [MaybeUninit::new(value); N],
        }
    }
}

impl<T: SimdInfo + Default + Copy, const N: usize> Default for SimdArray<T, N> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

// === Indexing ===

impl<T: SimdInfo, const N: usize> Index<usize> for SimdArray<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < N);
        unsafe { &self.data[index].assume_init_ref() }
    }
}

impl<T: SimdInfo, const N: usize> IndexMut<usize> for SimdArray<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < N);
        unsafe { self.data[index].assume_init_mut() }
    }
}

impl<T: SimdInfo, const N: usize> SimdArray<T, N> {
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> T {
        debug_assert!(index < N);
        unsafe { *self.data.get_unchecked(index).assume_init_ref() }
    }

    #[inline(always)]
    pub unsafe fn get_unchecked_mut(&mut self, index: usize) -> &mut T {
        debug_assert!(index < N);
        unsafe { self.data.get_unchecked_mut(index).assume_init_mut() }
    }
}

// === Utility Traits ===

impl<T: SimdInfo + Copy, const N: usize> Clone for SimdArray<T, N> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: SimdInfo + fmt::Debug, const N: usize> fmt::Debug for SimdArray<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(unsafe { self.data.assume_init_ref() })
            .finish()
    }
}

// === Simd Access ===

impl<T: SimdInfo, const N: usize> SimdArray<T, N> {
    #[inline(always)]
    pub fn load_simd(&self, index: usize) -> ArchSimd<T> {
        debug_assert!(index % T::LANES == 0);
        unsafe {
            let ptr = self.data.as_ptr().add(index) as *const T;
            std::ptr::read(ptr as *const ArchSimd<T>)
        }
    }

    #[inline(always)]
    pub fn store_simd(&mut self, index: usize, vec: ArchSimd<T>) {
        debug_assert!(index % T::LANES == 0);
        unsafe {
            let ptr = self.data.as_mut_ptr().add(index) as *mut T;
            std::ptr::write(ptr as *mut ArchSimd<T>, vec);
        }
    }

    #[inline(always)]
    pub fn partial_store_simd(&mut self, index: usize, vec: ArchSimd<T>, amount: usize) {
        debug_assert!(index + amount < N);
        let indices = Simd::<i32, { T::LANES }>::from_array(std::array::from_fn(|i| i as i32));
        let amounts = Simd::<i32, { T::LANES }>::splat(amount as i32);
        let mask = amounts.simd_gt(indices).cast::<<T as SimdElement>::Mask>();
        self.masked_store_simd(index, vec, mask);
    }

    #[inline(always)]
    pub fn masked_store_simd(&mut self, index: usize, vec: ArchSimd<T>, mask: Mask<<T as SimdElement>::Mask, { T::LANES }>) {
        debug_assert!(index < N);
        unsafe {
            let slice = std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr().add(index) as *mut T,
                T::LANES,
            );
            vec.store_select_unchecked(slice, mask);
        }
    }
}

// === Basic Operator Imeplementations ===

impl_simd_array_op!(Add, AddAssign, add, add_assign, +, +=);
impl_simd_array_op!(Sub, SubAssign, sub, sub_assign, -, -=);
impl_simd_array_op!(Mul, MulAssign, mul, mul_assign, *, *=);
impl_simd_array_op!(Div, DivAssign, div, div_assign, /, /=);

// === Additional Operations ===

impl<T: SimdInfo, const N: usize> SimdArray<T, N>
where
    [(); T::LANES]:,
{
    pub fn multiset_many<const M: usize>(arrays: &mut [&mut Self; M], values: &[T; M], mut index: usize, mut amount: isize) {
        let vecs: [ArchSimd<T>; M] = std::array::from_fn(|i| ArchSimd::<T>::splat(values[i]));

        let indices = Simd::<i32, {T::LANES}>::from_array(std::array::from_fn(|i| i as i32));
        while amount > 0 {
            let amounts = Simd::<i32, {T::LANES}>::splat(amount as i32);
            let mask = indices.simd_lt(amounts).cast::<<T as SimdElement>::Mask>();
            for i in 0..M {
                unsafe { arrays[i].masked_store_simd(index, *vecs.get_unchecked(i), mask); }
            }
            amount -= T::LANES as isize;
            index += T::LANES;
        }
    }
}

impl<T: SimdInfo + NumCast + Debug, const N: usize> SimdArray<T, N>
where
    [(); T::LANES]:,
    T: Add<Output = T>,
    T: Mul<Output = T>,
    ArchSimd<T>: Add<Output = ArchSimd<T>>,
    ArchSimd<T>: Mul<Output = ArchSimd<T>>,
    ArchSimd<T>: AddAssign,
{
    pub fn iota_custom(offset: T, increment: T) -> Self {
        let mut result = Self::new_uninit();

        let iota_array = std::array::from_fn(|i| NumCast::from(i).unwrap());
        let increment_vec = ArchSimd::splat(increment);
        let lanes_increment_vec = ArchSimd::splat(increment *  NumCast::from(T::LANES).unwrap());
        let iota_vec = ArchSimd::from_array(iota_array) * increment_vec;
        
        let mut cur_vec = ArchSimd::splat(offset) + iota_vec;
        
        result.store_simd(0, cur_vec);
        for i in (T::LANES..Self::TAIL_START).step_by(T::LANES) {
            cur_vec += lanes_increment_vec;
            result.store_simd(i, cur_vec);
        }

        if Self::HAS_TAIL {
            cur_vec += lanes_increment_vec;
            result.partial_store_simd(Self::TAIL_START, cur_vec, Self::TAIL_SIZE);
        }

        result
    }

    pub fn iota(offset: T) -> Self {
        let mut result = Self::new_uninit();

        let iota_array = std::array::from_fn(|i| NumCast::from(i).unwrap());
        let lanes_increment_vec = ArchSimd::splat(NumCast::from(T::LANES).unwrap());
        let iota_vec = ArchSimd::from_array(iota_array);
        
        let mut cur_vec = ArchSimd::splat(offset) + iota_vec;
        
        result.store_simd(0, cur_vec);
        for i in (T::LANES..Self::TAIL_START).step_by(T::LANES) {
            cur_vec += lanes_increment_vec;
            result.store_simd(i, cur_vec);
        }

        if Self::HAS_TAIL {
            cur_vec += lanes_increment_vec;
            result.partial_store_simd(Self::TAIL_START, cur_vec, Self::TAIL_SIZE);
        }

        result
    }
}

impl<T: SimdInfo + Float, const N: usize> SimdArray<T, N>
where
    [(); T::LANES]:,
    ArchSimd<T>: SimdFloat + StdFloat,
    T: Sub<Output = T>,
    ArchSimd<T>: Sub<Output = ArchSimd<T>>,
    ArchSimd<T>: Add<Output = ArchSimd<T>>,
{
    pub fn fract(&self) -> Self {
        let mut result = Self::new_uninit();
        for i in (0..Self::TAIL_START).step_by(T::LANES) {
            let data = self.load_simd(i);
            let new_data = data - data.floor();
            result.store_simd(i, new_data);
        }

        if Self::HAS_TAIL {
            let data = self.load_simd(Self::TAIL_START);
            let new_data = data - data.floor();
            result.partial_store_simd(Self::TAIL_START, new_data, Self::TAIL_SIZE);
        }

        result
    }
}

impl<T: SimdInfo + NumCast, const N: usize> SimdArray<T, N>
where
    [(); T::LANES]:,
    T: Mul<Output = T>,
    T: Add<Output = T>,
    T: Sub<Output = T>,
    ArchSimd<T>: Mul<Output = ArchSimd<T>>,
    ArchSimd<T>: Add<Output = ArchSimd<T>>,
    ArchSimd<T>: Sub<Output = ArchSimd<T>>,
    ArchSimd<T>: SimdFloat + StdFloat,
{
    pub fn quintic_lerp(&self) -> Self {
        let mut result = Self::new_uninit();

        let six = ArchSimd::splat(NumCast::from(6.0).unwrap());
        let ten = ArchSimd::splat(NumCast::from(10.0).unwrap());
        let neg_fifteen = ArchSimd::splat(NumCast::from(-15.0).unwrap());
        
        for i in (0..Self::TAIL_START).step_by(T::LANES) {
            let t = self.load_simd(i);
            let new_data = t * t * t * t.mul_add(t.mul_add(six, neg_fifteen), ten);
            result.store_simd(i, new_data);
        }

        if Self::HAS_TAIL {
            let t = self.load_simd(Self::TAIL_START);
            let new_data = t * t * t * t.mul_add(t.mul_add(six, neg_fifteen), ten);
            result.partial_store_simd(Self::TAIL_START, new_data, Self::TAIL_SIZE);
        }

        result
    }
}
