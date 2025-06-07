use std::ffi::CString;
use std::os::raw::c_float;
use thiserror::Error;
use xgb_sys::{
    BoosterHandle, XGBoosterCreate, XGBoosterFree, XGBoosterGetNumFeature, XGBoosterLoadModel,
    XGBoosterPredictFromDMatrix, XGBoosterSaveModel, XGBoosterSetParam, XGBoosterUpdateOneIter,
};

use crate::dmatrix::DMatrix;

#[derive(Error, Debug)]
pub enum XGBoostError {
    #[error("Cannot create Booster")]
    Create,
    #[error("Unable to load error")]
    Load,
    #[error("Cannot predict")]
    Predict,
    #[error("Iteration {0} broke")]
    Train(usize),
    #[error("Cannot set {0} to {1}")]
    Config(String, String),
    #[error("Cannot save model")]
    Save,
    #[error("Cannot get booster info: {0}")]
    GetInfo(String),
}

#[derive(Debug)]
pub struct Booster {
    handle: BoosterHandle,
}

impl Booster {
    pub fn new() -> Result<Self, XGBoostError> {
        let mut handle = std::ptr::null_mut();
        unsafe {
            if XGBoosterCreate(std::ptr::null_mut(), 0, &mut handle) == 0 {
                Ok(Booster { handle })
            } else {
                Err(XGBoostError::Create)
            }
        }
    }

    pub fn set_conf(&mut self, key: &str, value: &str) -> Result<(), XGBoostError> {
        let c_key = CString::new(key).unwrap();
        let c_value = CString::new(value).unwrap();
        unsafe {
            if XGBoosterSetParam(self.handle, c_key.as_ptr(), c_value.as_ptr()) == 0 {
                Ok(())
            } else {
                Err(XGBoostError::Config(key.to_string(), value.to_string()))
            }
        }
    }

    pub fn get_number_of_features(&self) -> Result<usize, XGBoostError> {
        let mut num_feats: u64 = u64::default();
        unsafe {
            if XGBoosterGetNumFeature(self.handle, &mut num_feats as *mut u64) == 0 {
                Ok(num_feats as usize)
            } else {
                Err(XGBoostError::GetInfo("Number of Features".to_string()))
            }
        }
    }

    pub fn train(
        dtrain: &DMatrix,
        _dtest: &DMatrix,
        num_boost: usize,
    ) -> Result<Self, XGBoostError> {
        let mut handle = std::ptr::null_mut();
        let booster = unsafe {
            if XGBoosterCreate([dtrain.handle].as_ptr(), 1, &mut handle) == 0 {
                Ok(Booster { handle })
            } else {
                Err(XGBoostError::Create)
            }
        }?;
        for i in 0..num_boost {
            unsafe {
                if XGBoosterUpdateOneIter(booster.handle, i as i32, dtrain.handle) != 0 {
                    return Err(XGBoostError::Train(i));
                }
            }
        }
        Ok(booster)
    }

    pub fn save_model(&self, fname: &str) -> Result<(), XGBoostError> {
        let fname = CString::new(fname).unwrap();
        unsafe {
            if XGBoosterSaveModel(self.handle, fname.as_ptr()) == 0 {
                Ok(())
            } else {
                Err(XGBoostError::Save)
            }
        }
    }

    pub fn load_model(&mut self, fname: &str) -> Result<(), XGBoostError> {
        let c_fname = CString::new(fname).unwrap();
        unsafe {
            if XGBoosterLoadModel(self.handle, c_fname.as_ptr()) == 0 {
                Ok(())
            } else {
                Err(XGBoostError::Load)
            }
        }
    }

    pub fn predict(&self, data: &DMatrix) -> Result<Vec<f32>, XGBoostError> {
        let mut out_result: *const c_float = std::ptr::null();
        let mut out_shape: u64 = 0;

        // Run the prediction
        let conf = CString::new("{\"training\": false, \"type\": 0, \"iteration_begin\": 0, \"iteration_end\": 0, \"strict_shape\": false}").expect("Cannot create pred config");
        unsafe {
            let predict_result = XGBoosterPredictFromDMatrix(
                self.handle,
                data.handle,
                conf.as_ptr(),
                &mut (&data.rows as *const u64) as *mut *const u64,
                &mut out_shape,
                &mut out_result,
            );

            if predict_result == 0 {
                // Convert the raw pointer to a slice and return the prediction result
                let slice = std::slice::from_raw_parts(out_result, data.rows as usize);
                Ok(slice.to_vec())
            } else {
                Err(XGBoostError::Predict)
            }
        }
    }
}

unsafe impl Sync for Booster { }

impl Drop for Booster {
    fn drop(&mut self) {
        unsafe {
            XGBoosterFree(self.handle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_booster_creation() {
        let booster = Booster::new();
        assert!(booster.is_ok(), "Failed to create Booster");
    }

    #[test]
    fn test_booster_config() {
        let mut booster = Booster::new().unwrap();
        let r = booster.set_conf("booster", "gblinear");
        assert!(r.is_ok(), "Could not set param");
    }

    #[test]
    fn test_booster_train_and_save() {
        let dtrain =
            DMatrix::try_from_data(&[0.1, 0.2, 0.3, 0.4], 2, 2).expect("Cannot create dtrain");
        let status = dtrain.try_add_label(&[1., 2.]);
        assert!(status.is_ok(), "Could not add label to train matrix");
        let dtest =
            DMatrix::try_from_data(&[0.1, 0.2, 0.3, 0.4], 2, 2).expect("Cannot create dtest");
        let booster = Booster::train(&dtrain, &dtest, 3).expect("Failed to train");
        let res = booster.save_model("yee.json");
        assert!(res.is_ok(), "Failed to save");
        let num_feats = booster.get_number_of_features().unwrap();
        assert_eq!(num_feats, 2, "Wrong number of features");
    }

    #[test]
    fn test_load_model() {
        let mut booster = Booster::new().expect("Failed to create Booster");

        // Replace "model.bin" with the path to your XGBoost model file
        let load_result = booster.load_model("yee.json");
        assert!(load_result.is_ok(), "Failed to load model");
    }

    #[test]
    fn test_predict() {
        let mut booster = Booster::new().expect("Failed to create Booster");

        // Replace "model.bin" with the path to your XGBoost model file
        booster
            .load_model("yee.json")
            .expect("Failed to load model");

        // Sample input data. Modify to match your model's input shape.

        let data = DMatrix::try_from_data(&[0.5, 1.2], 2, 1).unwrap();

        let prediction = booster.predict(&data);
        assert!(prediction.is_ok(), "Prediction failed");
        assert!(
            !prediction.unwrap().is_empty(),
            "Prediction result is empty"
        );
    }
}
