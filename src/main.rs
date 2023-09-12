use nash::*;
use ndarray::prelude::*;
use std::sync::mpsc::{Sender, Receiver};
use std::sync::mpsc;

const NUM_THREADS: u64 = 5;
const NUM_GENERATE: u64 = 100;
const NUM_REFINEMENTS: u64 = 3;

fn worker(seed: u64, ivan: Array<f32, Ix2>, opal: Array<f32, Ix2>, tx: Sender<Candidate>) {
    let cand = generate_candidate(seed, ivan, opal);
    tx.send(cand).unwrap();
}

fn main() {
    let base_seed: u64 = 824131567215;
    let mut ivan: Array<f32, Ix2> = Array::<f32, Ix2>::from_elem((2, 10), unflattener(&0.5));
    let mut opal: Array<f32, Ix2> = Array::<f32, Ix2>::from_elem((2, 10), unflattener(&0.5));

    for refinement in 0..NUM_REFINEMENTS
    {
        let (tx, rx): (Sender<Candidate>, Receiver<Candidate>) = mpsc::channel();

        for i in 0..NUM_THREADS {
            let txc = tx.clone();
            let ivan_clone = ivan.clone();
            let opal_clone = opal.clone();
            std::thread::spawn(move || {
                worker(base_seed + refinement*1000+i, ivan_clone, opal_clone, txc);
            });
        }

        let mut num_sent: u64 = NUM_THREADS;
        let mut num_recvd: u64 = 0;
        let mut o_candidates: Vec<Array<f32, Ix2>> = Vec::new();
        let mut i_candidates: Vec<Array<f32, Ix2>> = Vec::new();
        loop {
            if num_recvd < NUM_GENERATE {
                let cand: Candidate = rx.recv().unwrap();
                num_recvd += 1;
                println!("\ncost[{}, {}]:\n{:.7e}", refinement, num_recvd, cand.cost);
                println!("opal:\n{:.7e}", &cand.opal.t());
                println!("ivan:\n{:.7e}", &cand.ivan.t());
                o_candidates.push(cand.opal);
                i_candidates.push(cand.ivan);
            } else {
                break;
            }

            if num_sent < NUM_GENERATE {
                let txc = tx.clone();
                let ivan_clone = ivan.clone();
                let opal_clone = opal.clone();
                std::thread::spawn(move || {
                    worker(base_seed + refinement*1000 + num_sent, ivan_clone, opal_clone, txc);
                });
                num_sent += 1;
            }
        }

        let result = strategy_min_max(&o_candidates, &i_candidates);
        println!("\nRESULT[{}]:\ncost: {:.7e}", refinement, result.cost);
        println!("opal_indx: {}", result.min_max_indx_opal);
        println!("ivan_indx: {}", result.min_max_indx_ivan);

        ivan = i_candidates[result.min_max_indx_ivan].clone();
        opal = o_candidates[result.min_max_indx_opal].clone();
        println!("opal:\n{:.7e}", &opal.t());
        println!("ivan:\n{:.7e}", &ivan.t());

        ivan = ivan.map(unflattener);
        opal = opal.map(unflattener);
    }
}
