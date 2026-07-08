fn main() {
    // With on-device STT enabled, transcribe-cpp links reference BLAS (cblas_sgemm).
    // Its runtime SONAME (libblas.so.3) lives in a subdir that isn't on the default
    // loader path, so bake an rpath into the binary — no LD_LIBRARY_PATH needed.
    // Override the directory with RHEMA_BLAS_DIR (e.g. an OpenBLAS location).
    if std::env::var_os("CARGO_FEATURE_LOCAL_STT").is_some() {
        let dir = std::env::var("RHEMA_BLAS_DIR")
            .unwrap_or_else(|_| "/usr/lib/x86_64-linux-gnu/blas".to_string());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_LOCAL_STT");
    println!("cargo:rerun-if-env-changed=RHEMA_BLAS_DIR");

    tauri_build::build()
}
