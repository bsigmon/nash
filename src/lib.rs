use ndarray::prelude::*;
use rand::{Rng,SeedableRng};
use rand_distr::{Normal, Distribution};
use lazy_static::lazy_static;
use rand_chacha::ChaCha8Rng;

pub struct Candidate {
    pub opal: Array<f32, Ix2>,
    pub ivan: Array<f32, Ix2>,
    pub cost: f32,
}

pub struct MinMaxSoln {
    pub min_max_indx_opal: usize,
    pub min_max_indx_ivan: usize,
    pub cost: f32,
}

pub fn strategy_min_max(o_candidates: &Vec<Array<f32, Ix2>>, i_candidates: &Vec<Array<f32, Ix2>>) -> MinMaxSoln {
    let i_cnt = i_candidates.len();
    let o_cnt = o_candidates.len();
    let mut min_max_cost: f32 = f32::MAX;
    let mut min_max_indx_opal: usize = o_cnt;
    let mut min_max_indx_ivan: usize = i_cnt;

    for ivan_indx in 0..i_cnt {
        let ivan_test: &Array<f32, Ix2> = &i_candidates[ivan_indx];
        let mut max_cost: f32 = f32::MIN;
        let mut max_indx: usize = o_cnt;
        for opal_indx in 0..o_cnt {
            let opal_test: &Array<f32, Ix2> = &o_candidates[opal_indx];
            let cost: f32 = compute_cost(opal_test, ivan_test);
            if max_cost < cost {
                max_cost = cost;
                max_indx = opal_indx;
            }
        }
        assert!(max_indx < o_cnt);

        if min_max_cost > max_cost {
            min_max_cost = max_cost;
            min_max_indx_opal = max_indx;
            min_max_indx_ivan = ivan_indx;
        }
    }
    assert!(min_max_indx_ivan < i_cnt);

    MinMaxSoln {
        min_max_indx_opal,
        min_max_indx_ivan,
        cost: min_max_cost,
    }
}

pub fn generate_candidate(seed: u64, ivan: Array<f32, Ix2>, opal: Array<f32, Ix2>) -> Candidate {
    const LOOP_LIMIT: u16 = 126;
    const CANDIDATE_LIMIT: usize = 250;

    let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(seed);
    let mut ivan: Array<f32, Ix2> = ivan;
    let mut opal: Array<f32, Ix2> = opal;
    let mut ivan_sd: f32 = 1.0;
    let mut opal_sd: f32 = 1.0;
    let mut cost: f32 = 0.0;

    for _ in 0..LOOP_LIMIT {
        let mut o_deltas: Vec<Array<f32, Ix2>> = Vec::new();
        let mut o_candidates: Vec<Array<f32, Ix2>> = Vec::new();
        let n_ivan: Normal<f32> = Normal::new(0.0, ivan_sd).unwrap();
        let n_opal: Normal<f32> = Normal::new(0.0, opal_sd).unwrap();

        o_candidates.push(opal.map(flattener));
        o_deltas.push(Array::zeros((2, 10)));

        for _opal_indx in 1..CANDIDATE_LIMIT {
            let delta: Array<f32, Ix2> = {
                let mut delta: Array<f32, Ix2> = Array::zeros((2, 10));
                delta_init(&mut rng, &n_opal, &mut delta);
                delta
            };
            o_candidates.push((&opal + &delta).map(flattener));
            o_deltas.push(delta);
        }

        let mut i_deltas: Vec<Array<f32, Ix2>> = Vec::new();
        let mut i_candidates: Vec<Array<f32, Ix2>> = Vec::new();
        for ivan_indx in 0..CANDIDATE_LIMIT {
            if ivan_indx == 0 {
                i_candidates.push(ivan.map(flattener));
                i_deltas.push(Array::zeros((2, 10)));
            } else {
                let delta: Array<f32, Ix2> = {
                    let mut delta: Array<f32, Ix2> = Array::zeros((2, 10));
                    delta_init(&mut rng, &n_ivan, &mut delta);
                    delta
                };
                i_candidates.push((&ivan + &delta).map(flattener));
                i_deltas.push(delta);
            }
        }

        let cand = strategy_min_max(&o_candidates, &i_candidates);
        let sd_ivan_update = delta_rms(&i_deltas[cand.min_max_indx_ivan]);
        let sd_opal_update = delta_rms(&o_deltas[cand.min_max_indx_opal]);
        ivan_sd = exp_moving_av_5(ivan_sd, sd_ivan_update);
        opal_sd = exp_moving_av_5(opal_sd, sd_opal_update);
        ivan = &ivan + &i_deltas[cand.min_max_indx_ivan];
        opal = &opal + &o_deltas[cand.min_max_indx_opal];
        cost = cand.cost;
    }
    Candidate {
        opal: opal.map(flattener),
        ivan: ivan.map(flattener),
        cost,
    }
}

lazy_static! {
    static ref SHOWDOWN_FILTER: Array<f32, Ix2> = { array![
        [0.0,   1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0],
        [-1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0],
        [-1.0, -1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0],
        [-1.0, -1.0, -1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0,  0.0,  1.0,  1.0,  1.0,  1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0,  0.0,  1.0,  1.0,  1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  0.0,  1.0,  1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  0.0,  1.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0,  0.0, 1.0],
        [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0]]
    };
}

lazy_static! {
    // The ace to 5 game as described in "Play Optimal Poker" by Andrew Brokos didn't allow the
    // players to have the same card.  By computing cost as 0 even for folds when the players have
    // the same card, we mathematically achieve this same effect.
    static ref DECK_FILTER: Array<f32, Ix2> = { array![
        [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]
    };
}

fn cost_showdown(opal: &Array<f32, Ix2>, ivan : &Array<f32, Ix2>, control: &mut Array<f32, Ix2>) -> f32 {
    let p: Array<f32, Ix2> = opal.dot(&ivan.t());
    *control += &p;
    let cost = (&*SHOWDOWN_FILTER * &p).sum();
    cost
}

fn cost_fold(opal: &Array<f32, Ix2>, ivan : &Array<f32, Ix2>, control: &mut Array<f32, Ix2>) -> f32 {
    //println!("cost_fold:\n{:?}\n{:?}", opal, ivan);
    let p: Array<f32, Ix2> = opal.dot(&ivan.t());
    *control += &p;
    let cost = (&*DECK_FILTER * &p).sum();
    cost
}

fn one_minus(x: &f32) -> f32 {
    1.0 - *x
}

fn identity(x: &f32) -> f32 {
    *x
}

pub fn compute_cost(opal: &Array<f32, Ix2>, ivan : &Array<f32, Ix2>) -> f32 {
    let opal_bets= opal.slice(s![0, ..]).into_shape((10, 1)).unwrap().map(identity);
    let ivan_calls = ivan.slice(s![0, ..]).into_shape((10, 1)).unwrap().map(identity);
    let ivan_bets = ivan.slice(s![1, ..]).into_shape((10, 1)).unwrap().map(identity);
    let opal_calls = opal.slice(s![1, ..]).into_shape((10, 1)).unwrap().map(identity);

    let opal_check_call = &opal_bets.map(one_minus) * &opal_calls;
    let opal_check_fold = &opal_bets.map(one_minus) * &opal_calls.map(one_minus);

    let mut cost = 0.0;
    let mut control_sum: Array<f32, Ix2> = Array::zeros((10,10).f());

    // Opal checks, Ivan checks: showdown, winner wins one chip
    cost += cost_showdown(&opal_bets.map(one_minus), &ivan_bets.map(one_minus), &mut control_sum);

    // Opal checks, Ivan bets, Opal folds: Ivan wins one chip
    cost -= cost_fold(&opal_check_fold, &ivan_bets, &mut control_sum);

    // Opal checks, Ivan bets, Opal calls: showdown, winner wins two chips
    cost += 2.0 * cost_showdown(&opal_check_call, &ivan_bets, &mut control_sum);

    // Opal bets, Ivan folds: Opal wins one chip
    cost += cost_fold(&opal_bets, &ivan_calls.map(one_minus), &mut control_sum);

    // Opal bets, Ivan calls: showdown, winner wins two chips
    cost += 2.0 * cost_showdown(&opal_bets, &ivan_calls, &mut control_sum);

    let mut control = control_sum.map(one_minus);
    let control_clone = control.clone();
    control *= &control_clone;
    assert!(control.sum() < 0.0001);
    cost
}

pub fn flattener(x: &f32) -> f32 {
    let flattened = libm::sinf(*x);
    flattened * flattened
}

pub fn unflattener(x: &f32) -> f32 {
    let unflattened = libm::sqrtf(*x);
    libm::asinf(unflattened)
}

pub fn delta_init<R: Rng + ?Sized>(rng: &mut R, n: &Normal<f32>, delta: &mut Array<f32, Ix2>) {
    let s = delta.shape();
    let rows = s[0];
    let cols = s[1];

    for i in 0..rows {
        for j in 0..cols {
            delta[[i, j]] = n.sample(rng);
        }
    }
}

pub fn exp_moving_av_5(old: f32, update: f32) -> f32 {
    const FACTOR: f32 = 0.8705505632961241;
    update + (old - update) * FACTOR
}

pub fn delta_rms(delta: &Array<f32, Ix2>) -> f32 {
    let cnt =  delta.len() as f32;
    libm::sqrtf(delta.map(|x| { x * x }).sum() / cnt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cost_showdown() {
        let i: Array1<f32> = array![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let o: Array1<f32> = array![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let i = i.into_shape((10, 1)).unwrap();
        let o = o.into_shape((10, 1)).unwrap();

        let mut control: Array<f32, Ix2> = Array::zeros((10,10));
        let c = cost_showdown(&o, &i, &mut control);
        assert_eq!(c, 1.0);
        assert_eq!(control.sum(), 1.0);

        let o: Array1<f32> = array![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let o = o.into_shape((10, 1)).unwrap();
        let mut control: Array<f32, Ix2> = Array::zeros((10,10));
        let c = cost_showdown(&o, &i, &mut control);
        assert_eq!(c, 2.0);
        assert_eq!(control.sum(), 2.0);

        let i: Array1<f32> = array![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let o: Array1<f32> = array![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let i = i.into_shape((10, 1)).unwrap();
        let o = o.into_shape((10, 1)).unwrap();

        let mut control: Array<f32, Ix2> = Array::zeros((10,10));
        let c = cost_showdown(&o, &i, &mut control);
        assert_eq!(c, -4.0);
        assert_eq!(control.sum(), 4.0);
    }

    #[test]
    fn test_cost_fold() {
        let i: Array1<f32> = array![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let o: Array1<f32> = array![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let i = i.into_shape((10, 1)).unwrap();
        let o = o.into_shape((10, 1)).unwrap();
        let mut control: Array<f32, Ix2> = Array::zeros((10,10));

        let c = cost_fold(&o, &i, &mut control);
        assert_eq!(c, 1.0);
        assert_eq!(control.sum(), 1.0);

        let o: Array1<f32> = array![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let o = o.into_shape((10, 1)).unwrap();
        let mut control: Array<f32, Ix2> = Array::zeros((10,10));
        let c = cost_fold(&o, &i, &mut control);
        assert_eq!(c, 2.0);
        assert_eq!(control.sum(), 2.0);

        let i: Array1<f32> = array![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let o: Array1<f32> = array![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let i = i.into_shape((10, 1)).unwrap();
        let o = o.into_shape((10, 1)).unwrap();
        let mut control: Array<f32, Ix2> = Array::zeros((10,10));

        let c = cost_fold(&o, &i, &mut control);
        assert_eq!(c, 5.0);
        assert_eq!(control.sum(), 6.0);
    }

    #[test]
    fn test_compute_cost() {
        let ivan: Array<f32, Ix2> = array![
            [9.9989468e-01, 9.9983245e-01, 9.9966949e-01, 9.9912906e-01, 9.9643213e-01, 2.2897455e-01, 1.0609054e-02, 5.2126111e-03, 3.1933694e-03, 4.7110589e-04],
            [9.9972153e-01, 9.9970138e-01, 9.9964488e-01, 9.9942535e-01, 9.9445516e-01, 6.5370405e-04, 4.3159263e-04, 6.6991179e-04, 9.9532264e-01, 9.9923706e-01]
        ];

        let opal: Array<f32, Ix2> = array![
            [9.7439247e-01, 9.4900459e-01, 5.6640947e-01, 5.9687842e-02, 2.8187014e-02, 1.1011532e-03, 5.2816287e-04, 7.7047752e-04, 3.7089421e-03, 9.8975599e-01],
            [9.9938011e-01, 9.9955279e-01, 9.9973500e-01, 9.9976361e-01, 9.9943525e-01, 9.9116480e-01, 9.7134686e-01, 6.5346025e-02, 7.5128523e-04, 1.2634672e-03]
        ];
        let cost = compute_cost(&opal, &ivan);
        assert!(cost < -4.246630);
        assert!(cost > -4.246640);

        // Book solution
        let ivan: Array<f32, Ix2> = array![
            [1.0, 1.0, 1.0, 1.0, 0.83, 0.59, 0.42, 0.16, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0]
        ];

        let opal: Array<f32, Ix2> = array![
            [0.81, 0.73, 0.63, 0.57, 0.0, 0.0, 0.0, 0.0, 0.36, 0.56],
            [1.0, 1.0, 1.0, 1.0, 1.0, 0.49, 0.42, 0.4, 0.0, 0.0]
        ];
        let cost = compute_cost(&opal, &ivan);
        // println!("cost: {}", cost);
        assert!(cost < -5.499);
        assert!(cost > -5.501);
    }

    #[test]
    fn test_delta_init() {
        let mut delta: Array<f32, Ix2> = Array::zeros((2, 10));
        let mut rng: ChaCha8Rng = ChaCha8Rng::seed_from_u64(124);
        let normal: Normal<f32> = Normal::new(0.0, 1.0).unwrap();

        delta_init(& mut rng, &normal, &mut delta);
        // println!("delta:\n {}", delta);
        assert!(delta[[0, 0]] < -0.639566);
        assert!(delta[[0, 0]] > -0.639568);
    }

    #[test]
    fn test_exp_moving_av_5() {
        let mut x: f32 = 0.75;
        x = exp_moving_av_5(x, 0.25);
        x = exp_moving_av_5(x, 0.25);
        x = exp_moving_av_5(x, 0.25);
        x = exp_moving_av_5(x, 0.25);
        x = exp_moving_av_5(x, 0.25);
        assert!(x > 0.4999);
        assert!(x < 0.5001);
    }

}
