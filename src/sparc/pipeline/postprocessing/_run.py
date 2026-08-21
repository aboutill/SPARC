import os
import logging
import json
import pydicom
import datetime
import shutil

import nibabel as nib
import numpy as np

from pydicom.dataset import Dataset, validate_file_meta
from pydicom.uid import generate_uid
from pydicom.datadict import DicomDictionary, keyword_dict
from pydicom.dataelem import DataElement


def set_private_dcm_tags(self, output_dir):
    """Register this pipeline's custom private DICOM tags (phase/slice
    number and counts) into pydicom's global tag dictionary, and
    optionally save their definitions as a JSON reference file.
    """
    
    # Define private Dicom fields
    private_dcm_fields = {
        0x20011008: ("US", "1", "Phase Number", "", "PhaseNumber"),
        0x2001100a: ("US", "1", "Slice Number MR", "", "SliceNumberMR"),
        0x20011017: ("US", "1", "Number Of Phases MR", "", "NumberOfPhasesMR"),
        0x20011018: ("US", "1", "Number Of Slices MR", "", "NumberOfSlicesMR"),
    }
    
    # Update the dictionary itself
    DicomDictionary.update(private_dcm_fields)
    
    # Update the reverse mapping from name to field
    private_dcm_keyword = dict([
        (val[4], hex(field)) for field, val in
        private_dcm_fields.items()])
    keyword_dict.update(private_dcm_keyword)
    
    if self.save_dcm_private_tags:
        # Save private Dicom keywords as json file
        json_path = os.path.join(output_dir, "private-dicom-tags.json")
        with open(json_path, "w") as json_file:
            json.dump(private_dcm_keyword, json_file, indent=4)
            

def nii2dcm(
        self,
        cine_nii_path,
        stack_infos,
        output_dir,
    ):
    """Convert a reconstructed cine NIfTI volume into a classic DICOM
    series (one file per slice/cardiac-phase combination), copying
    acquisition metadata from a source stack's DICOM header.
    """
    
    # Dicom fields to copy from target stack
    dcm_fields_to_copy = [
        "SpecificCharacterSet",
        "ImageType",
        "SOPClassUID",
        "SOPInstanceUID",
        "Modality",
        "Manufacturer",
        "StudyDescription",
        "ProcedureCodeSequence",
        "RelatedSeriesSequence",
        "PatientName",
        "PatientID",
        "PatientSize",
        "PatientIdentityRemoved",
        "BodyPartExamined",
        "ScanningSequence",
        "SequenceVariant",
        "ScanOptions",
        "AngioFlag",
        "RepetitionTime",
        "EchoTime",
        "NumberOfAverages",
        "ImagingFrequency",
        "ImagedNucleus",
        "MagneticFieldStrength",
        "EchoTrainLength",
        "DeviceSerialNumber",
        "SoftwareVersions",
        "BeatRejectionFlag",
        "ReceiveCoilName",
        "TransmitCoilName",
        "FlipAngle",
        "SAR",
        "PatientPosition",
        "PositionReferenceIndicator",
        "PhotometricInterpretation",
    ]
    
    #
    if self.ImplementationVersionName is None:
        self.ImplementationVersionName = "Default"
    if self.ProtocolName is None:
        self.ProtocolName = "Default"
    if self.StudyDescription is None:
        self.StudyDescription = "Default"
    if self.SeriesDescription is None:
        self.SeriesDescription = "Default"
    
    # Load nifti cine vol
    cine_nii = nib.load(cine_nii_path)
    cine = cine_nii.get_fdata()
    
    # Extract image information to populate Dicom header
    # Get nifti pixdim, resolution, and affine
    dim = cine_nii.header["dim"][1:5]
    res = cine_nii.header["pixdim"][1:5]
    aff = cine_nii.affine
    
    # Dicom and nifti X/Y axis are swapped and in opposite direction
    # See nibabel (https://nipy.org/nibabel/dicom/dicom_orientation.html)
    dim[0], dim[1] = dim[1], dim[0]
    res[0], res[1] = res[1], res[0]
    aff[:2,:] *= -1
    aff = aff.dot([[0,1,0,0], [1,0,0,0], [0,0,1,0], [0,0,0,1]])
    
    # Modify cine nifti vol to Dicom standard
    cine = np.swapaxes(cine, 0, 1)
    cine = np.clip(np.round(cine), 0, 4095).astype(np.uint16)
    
    # Derivation of cosine direction and slice transform location 
    cosine_dir_x = aff[:3,0] / res[0]
    cosine_dir_y = aff[:3,1] / res[1]
    slice_transform = lambda n: aff.dot([[0], [0], [n], [1]])
    
    # Set RR interval, heart-rate, trigger time
    rr_interval = res[3] * dim[3] # ms
    heart_rate = 60 * 1e3 / rr_interval # beats/min
    trigger_times = np.linspace(0, rr_interval, num=dim[3])
    
    # Set frame index
    frame_ind = np.arange(dim[3])
    
    # Set slice index
    slice_ind = np.arange(dim[2])
    
    # Set slice location 
    slice_loc = [slice_transform(ind) for ind in slice_ind]
    
    # Intensity extrema
    min_int = int(np.min(cine))
    max_int = int(np.max(cine))
    
    # Set intensity window
    window_width = max_int - min_int
    window_center = min_int + window_width / 2
    
    # Load 1st serie dicom
    dcm_path = stack_infos[0]["DICOM"]
    dcm = pydicom.dcmread(dcm_path)
    
    # Get time
    time = datetime.datetime.now()
    
    # Create dicom file meta
    file_meta = Dataset()
    file_meta.FileMetaInformationVersion = b"\x00\x01"
    file_meta.MediaStorageSOPClassUID = dcm.file_meta.MediaStorageSOPClassUID
    file_meta.TransferSyntaxUID = dcm.file_meta.TransferSyntaxUID
    file_meta.ImplementationClassUID = generate_uid()
    file_meta.ImplementationVersionName = self.ImplementationVersionName

    # Create dicom cine vol
    cine_dcm = Dataset()
    
    # Set cine vol specific dicom fields
    cine_dcm.InstanceCreationDate = time.strftime("%Y%m%d")
    cine_dcm.InstanceCreationTime = time.strftime("%H%M%S.%f") 
    cine_dcm.InstanceCreatorUID = generate_uid()
    cine_dcm.StudyDate = time.strftime("%Y%m%d")
    cine_dcm.SeriesDate = time.strftime("%Y%m%d")
    cine_dcm.AcquisitionDate = time.strftime("%Y%m%d")
    cine_dcm.ContentDate = time.strftime("%Y%m%d")
    cine_dcm.StudyTime = time.strftime("%H%M%S.%f") 
    cine_dcm.SeriesTime = time.strftime("%H%M%S.%f") 
    cine_dcm.AcquisitionTime = time.strftime("%H%M%S.%f") 
    cine_dcm.ContentTime = time.strftime("%H%M%S.%f") 
    
    # Copy dicom fields from target stack
    for dcm_field in dcm_fields_to_copy:
        if not hasattr(dcm, dcm_field):
            # Log if missing dcm field
            logging.info(
                f"No {dcm_field} field found in"
                f" dicom file {dcm_path}."
            )
        else:
            # Copy dcm field (VR and value)
            vr = dcm[dcm_field].VR
            val = dcm[dcm_field].value
            elem = DataElement(dcm_field, vr, val)
            cine_dcm.add(elem)

    # Set remaining dicom fields
    # Dicom description
    cine_dcm.ProtocolName = self.ProtocolName
    cine_dcm.StudyDescription = self.StudyDescription
    cine_dcm.SeriesDescription = self.SeriesDescription

    # Image attributes
    cine_dcm.MRAcquisitionType = "3D"
    cine_dcm.SequenceName = ""
    cine_dcm.SliceThickness = f"{res[2]:.8f}"
    cine_dcm.SpacingBetweenSlices = f"{res[2]:.8f}"
    cine_dcm.NumberOfPhaseEncodingSteps = int(dim[0])
    cine_dcm.PercentSampling = ""
    cine_dcm.PercentPhaseFieldOfView = ""
    cine_dcm.PixelBandwidth = ""
    cine_dcm.NominalInterval = str(int(rr_interval)) #ms
    cine_dcm.LowRRValue = ""
    cine_dcm.HighRRValue = ""
    cine_dcm.IntervalsAcquired = ""
    cine_dcm.IntervalsRejected = ""
    cine_dcm.HeartRate = str(int(heart_rate))
    cine_dcm.AcquisitionMatrix = [
        int(dim[0]), 
        int(dim[1]), 
        int(dim[2]), 
        int(dim[3]),
    ]
    cine_dcm.InPlanePhaseEncodingDirection = "ROW" 
    cine_dcm.StudyInstanceUID = generate_uid()
    cine_dcm.SeriesInstanceUID = generate_uid()
    cine_dcm.StudyID = ""
    cine_dcm.SeriesNumber = ""
    cine_dcm.AcquisitionNumber = ""
    cine_dcm.ImageOrientationPatient = [
        f"{cosine_dir_y[0]:.8f}", 
        f"{cosine_dir_y[1]:.8f}", 
        f"{cosine_dir_y[2]:.8f}", 
        f"{cosine_dir_x[0]:.8f}",
        f"{cosine_dir_x[1]:.8f}",
        f"{cosine_dir_x[2]:.8f}",
    ]
    cine_dcm.FrameOfReferenceUID = generate_uid()
    cine_dcm.NumberOfTemporalPositions = str(dim[3])

    # Image attributes (cont.)
    cine_dcm.SamplesPerPixel = 1
    cine_dcm.Rows = int(dim[0])
    cine_dcm.Columns = int(dim[1])
    cine_dcm.PixelSpacing = [f"{res[0]:.8f}", f"{res[1]:.8f}"]
    cine_dcm.BitsAllocated = 16
    cine_dcm.BitsStored = 12
    cine_dcm.HighBit = 11
    cine_dcm.PixelRepresentation = 0
    cine_dcm.SmallestImagePixelValue = min_int
    cine_dcm.LargestImagePixelValue = max_int
    cine_dcm.WindowCenter = f"{window_center:.8f}"
    cine_dcm.WindowWidth = f"{window_width:.8f}"
    cine_dcm.RescaleIntercept = "0.0"
    cine_dcm.RescaleSlope = "1.0"
    cine_dcm.LossyImageCompression = "00"
    cine_dcm.PresentationLUTShape = "IDENTITY"

    # Private Dicom fields
    cine_dcm.NumberOfSlicesMR = int(dim[2])
    cine_dcm.NumberOfPhasesMR = int(dim[3])
    
    # Update SOP instance UID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    cine_dcm.SOPInstanceUID = generate_uid()
    
    # Append file meta to dicom
    cine_dcm.file_meta = file_meta
    cine_dcm.is_implicit_VR = False
    cine_dcm.is_little_endian = True
    
    # Validate file meta
    validate_file_meta(cine_dcm.file_meta, enforce_standard=True)
    
    # Iterate over slice and frames
    for i in range(dim[2]):
        for j in range(dim[3]):
        
            # Spatio-temporal index
            k = i*dim[3] + j
            
            # Update frame specific dicom fields
            cine_dcm.TriggerTime = f"{trigger_times[j]:.8f}"
            cine_dcm.InstanceNumber = str(k)
            cine_dcm.ImagePositionPatient = [
                f"{slice_loc[i][0][0]:.8f}", 
                f"{slice_loc[i][1][0]:.8f}",
                f"{slice_loc[i][2][0]:.8f}",
            ]
            cine_dcm.SliceLocation = f"{slice_loc[i][2][0]:.8f}"
            
            # Update frame specific private dicom fields
            cine_dcm.PhaseNumber = int(frame_ind[j])
            cine_dcm.SliceNumberMR = int(slice_ind[i])
        
            # Create pixel data
            cine_dcm.PixelData = cine[:,:,i,j].tobytes()
            
            # Save dicom
            img_path = f"img-{k:06d}.dcm"
            output_path = os.path.join(output_dir, img_path)
            cine_dcm.save_as(output_path, enforce_file_format=True)
    

def run(
        self,
        cine_nii_path,
        stack_infos,
        output_dir,
        file_prefix=None,
        profile=False,
        debug=False,
    ):
    """Convert a reconstructed cine NIfTI volume into a DICOM series."""
    
    logging.info(f"Converting reconstructed 3D+time volume {cine_nii_path}...")
    
    if profile or debug:
        start_time = datetime.datetime.now()
        
    # Get absolute path
    cine_nii_path = os.path.abspath(cine_nii_path)
    output_dir = os.path.abspath(output_dir)
    
    # Initialize output dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Set private dicom fields
    self.set_private_dcm_tags(output_dir)
    
    # Set output dir
    name = "cine"
    dirname = f"{file_prefix}_{name}" if file_prefix is not None else name
    dcm_dir = os.path.join(output_dir, dirname)
    os.makedirs(dcm_dir, exist_ok=True)
    
    # Create dicom
    self.nii2dcm(
        cine_nii_path=cine_nii_path,
        stack_infos=stack_infos,
        output_dir=dcm_dir,
    )
                    
    if self.zip_dcm_files:
        # Set path 
        dirname =  f"{file_prefix}_{name}" if file_prefix is not None else name
        zip_path = os.path.join(output_dir, dirname)
        
        # Zip folder
        shutil.make_archive(zip_path, "zip", dcm_dir)
        
        # Remove folder
        shutil.rmtree(dcm_dir)
        
    if profile or debug:
        elapsed_time = datetime.datetime.now() - start_time
        logging.info(f"Time for NIfTI to DICOM conversion: {elapsed_time}")
        
    return dcm_dir