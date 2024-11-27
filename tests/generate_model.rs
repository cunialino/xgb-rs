use rand::Rng;
use xgb_rs::booster::Booster;
use xgb_rs::dmatrix::DMatrix;

const N_ROWS: usize = 10000;
const N_COLS: usize = 30;

#[test]
fn test_model_generation() {
    let data: Vec<f32> = (0..N_COLS * N_ROWS)
        .map(|_| rand::thread_rng().gen())
        .collect();
    let target: Vec<f32> = (0..N_ROWS).map(|_| rand::thread_rng().gen()).collect();
    let dmat =
        DMatrix::try_from_data(data.as_slice(), N_ROWS as u64, N_COLS as u64).expect("Failed dmat");
    dmat.try_add_label(target.as_slice())
        .expect("Could not set target");
    let booster = Booster::train(&dmat, &dmat, 700).expect("Could not train");
    booster
        .save_model("silly_model.json")
        .expect("Could not save model");
}
