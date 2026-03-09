#[cfg(feature = "metal")]
fn main() {
    use std::env;
    use std::path::PathBuf;
    use std::process::Command;

    println!("cargo:rerun-if-changed=metal/sky98_metal.swift");

    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR missing"));
    let helper_path = out_dir.join("sky98-metal-helper");
    let module_cache = out_dir.join("swift-module-cache");
    let source = PathBuf::from("metal/sky98_metal.swift");

    std::fs::create_dir_all(&module_cache).expect("create swift module cache");

    let status = Command::new("xcrun")
        .args([
            "swiftc",
            "-O",
            "-module-cache-path",
        ])
        .arg(&module_cache)
        .arg(&source)
        .arg("-o")
        .arg(&helper_path)
        .status()
        .expect("failed to invoke swiftc");

    if !status.success() {
        panic!("failed to build Metal helper with swiftc");
    }

    println!("cargo:rustc-env=SKY98_METAL_HELPER={}", helper_path.display());
}

#[cfg(not(feature = "metal"))]
fn main() {}
