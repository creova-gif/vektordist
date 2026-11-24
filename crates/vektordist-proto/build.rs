fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "codegen")]
    {
        tonic_build::configure()
            .build_server(true)
            .build_client(true)
            .out_dir("src")
            .compile_protos(&["proto/vektordist.proto"], &["proto"])?;
    }
    Ok(())
}
