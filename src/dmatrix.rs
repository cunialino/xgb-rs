use std::{ffi::CString, os::raw::c_void};
use thiserror::Error;
use xgb_sys::{XGDMatrixCreateFromMat, XGDMatrixFree, XGDMatrixSetFloatInfo};

#[derive(Error, Debug)]
pub enum DMatrixError {
    #[error("Cannot create DMatrix")]
    Create,
}

pub struct DMatrix {
    pub(crate) handle: *mut c_void,
    pub(crate) rows: u64,
    _cols: u64,
}

impl DMatrix {
    pub fn try_new() -> Result<Self, DMatrixError> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        unsafe {
            if XGDMatrixCreateFromMat(std::ptr::null(), 0, 0, f32::NAN, &mut handle) == 0 {
                Ok(DMatrix {
                    handle,
                    rows: 0,
                    _cols: 0,
                })
            } else {
                Err(DMatrixError::Create)
            }
        }
    }

    pub fn try_from_data(data: &[f32], rows: u64, cols: u64) -> Result<Self, DMatrixError> {
        let mut handle: *mut c_void = std::ptr::null_mut();
        unsafe {
            if XGDMatrixCreateFromMat(data.as_ptr(), rows, cols, f32::NAN, &mut handle) == 0 {
                Ok(DMatrix { handle, rows, _cols: cols })
            } else {
                Err(DMatrixError::Create)
            }
        }
    }

    pub fn try_add_label(&self, data: &[f32]) -> Result<(), DMatrixError> {
        let lab = CString::new("label").map_err(|_| DMatrixError::Create)?;
        unsafe {
            if XGDMatrixSetFloatInfo(self.handle, lab.as_ptr(), data.as_ptr(), self.rows) == 0 {
                Ok(())
            }
            else {
                Err(DMatrixError::Create)
            }
        }
    }
}

impl Drop for DMatrix {
    fn drop(&mut self) {
        unsafe {
            XGDMatrixFree(self.handle);
        }
    }
}
