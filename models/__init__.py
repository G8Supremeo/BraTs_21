from models.spectral_3d_unet_afw_attention import UNet3D_SpectralAFW

MODEL_REGISTRY = {
    "enhanced_spectral_3d_unet_afw_attention": UNet3D_SpectralAFW,
    "unet3d_spectral_afw": UNet3D_SpectralAFW,  # Backward compatibility
}
