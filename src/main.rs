mod random;

use random::Random;

fn main() {

    let rand = Random::new(0);
    
    for i in 0..10 {
        let value = rand.mix_u64(i);
        println!("{i}: {:064b}", value);
    }
}
