use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let target = env::var("TARGET").expect("Could not get TARGET environment variable");
    let out_dir = env::var("OUT_DIR").expect("Could not get OUT_DIR environment variable");
    let xgb_root = Path::new("xgboost");
    // Ensure the file is writable by removing any existing version
    let (stdcpp, gomp) = if target.contains("apple") {
        ("c++", "omp")
    } else {
        ("stdc++", "gomp")
    };

    // Build XGBoost with CMake as a static library
    let xgb_root = xgb_root
        .canonicalize()
        .expect("Failed to canonicalize xgb_root");
    let xgb_dest = Config::new(&xgb_root)
        .define("BUILD_STATIC_LIB", "ON") // Build as static library
        .define("USE_OPENMP", "ON") // Enable OpenMP support
        .uses_cxx11()
        .build();

    // Generate Rust bindings with bindgen
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", xgb_root.join("include").display()))
        .clang_arg(format!(
            "-I{}",
            xgb_root.join("dmlc-core/include").display()
        ))
        .clang_arg("-fopenmp")
        .generate()
        .expect("Unable to generate bindings.");

    let out_path = PathBuf::from(out_dir.clone());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    println!("cargo:rustc-link-search=native=/usr/lib");

    println!("cargo:rustc-link-lib={}", stdcpp);
    println!("cargo:rustc-link-lib=dylib={}", gomp);

    println!("cargo:rustc-link-search={}", xgb_dest.join("lib").display());
    println!(
        "cargo:rustc-link-search={}",
        xgb_dest.join("lib64").display()
    );
    // Link XGBoost and its dependencies
    println!("cargo:rustc-link-lib=static=xgboost");
    println!("cargo:rustc-link-lib=static=dmlc");

    // Force rebuild if these files change
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=xgboost");
}
