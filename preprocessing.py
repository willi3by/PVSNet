import ants

def preproc_for_pvsnet(nii_brain_file, ref_brain_file):
    nii = ants.image_read(nii_brain_file)
    ref_nii = ants.image_read(ref_brain_file)
    nii_allin = ants.registration(fixed=ref_nii,
                                  moving=nii,
                                  type_of_transform="Rigid")
    nii_resamp = nii.resample_image((512,512,32), True, 0)
    nii_scaled = (nii_resamp - nii_resamp.mean())/nii_resamp.std()
    nii_windowed = nii_scaled.iMath_truncate_intensity(0.82, 0.9)
    nii_trimmed = nii_windowed.crop_indices((0,0,8), (512,512,24))
    return nii_trimmed