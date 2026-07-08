fn main() {
    // transcribe-cpp's CPU decoder (parakeet/gigaam) calls `cblas_sgemm` but links no
    // BLAS itself, so the final binary needs one. Only required for the `local-stt`
    // feature; the default cloud build links nothing extra.
    //
    // Override the BLAS library via RHEMA_BLAS_LIB (e.g. "openblas" for speed) and
    // its search dir via RHEMA_BLAS_DIR if it isn't on the default linker path.
    if std::env::var_os("CARGO_FEATURE_LOCAL_STT").is_some() {
        let lib = std::env::var("RHEMA_BLAS_LIB").unwrap_or_else(|_| "blas".to_string());
        if let Ok(dir) = std::env::var("RHEMA_BLAS_DIR") {
            println!("cargo:rustc-link-search=native={dir}");
        }
        println!("cargo:rustc-link-lib={lib}");
    }
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_LOCAL_STT");
    println!("cargo:rerun-if-env-changed=RHEMA_BLAS_LIB");
    println!("cargo:rerun-if-env-changed=RHEMA_BLAS_DIR");
}
