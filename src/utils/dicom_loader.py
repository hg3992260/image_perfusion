import os
import pydicom
import numpy as np
from collections import defaultdict
import concurrent.futures

class DicomSeries:
    def __init__(self, series_uid, file_paths):
        self.series_uid = series_uid
        self.file_paths = file_paths
        self.description = ""
        self.modality = ""
        self.patient_name = ""
        self.study_date = ""
        self.dims = (0, 0, 0, 0) # x, y, z, t
        self.pixel_array = None # 4D array
        self.metadata = {}
        
        # Load basic info from first file (metadata only)
        if file_paths:
            try:
                ds = pydicom.dcmread(file_paths[0], stop_before_pixels=True)
                self.description = getattr(ds, 'SeriesDescription', 'Unknown')
                self.modality = getattr(ds, 'Modality', 'Unknown')
                self.patient_name = str(getattr(ds, 'PatientName', 'Unknown'))
                self.study_date = getattr(ds, 'StudyDate', 'Unknown')
                self.metadata['TR'] = getattr(ds, 'RepetitionTime', 1000.0) / 1000.0
                self.metadata['TE'] = getattr(ds, 'EchoTime', 0) / 1000.0
            except:
                pass
            
    def load_data(self):
        """
        Loads and sorts DICOM files into a 4D numpy array (x, y, z, t).
        Uses multithreading for faster IO.
        """
        if not self.file_paths:
            return None

        # 1. Read Metadata in Parallel
        # We need to sort files to determine structure.
        # Reading headers is fast but for many files it adds up.
        
        def read_metadata(path):
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
                return {
                    'path': path,
                    'in': int(getattr(ds, 'InstanceNumber', 0)),
                    'at': float(getattr(ds, 'AcquisitionTime', 0) or 0),
                    'sl': float(getattr(ds, 'SliceLocation', 0) or getattr(ds, 'ImagePositionPatient', [0,0,0])[2]),
                    'rows': ds.Rows,
                    'cols': ds.Columns,
                    'slope': getattr(ds, 'RescaleSlope', 1),
                    'intercept': getattr(ds, 'RescaleIntercept', 0)
                }
            except Exception as e:
                # print(f"Error reading {path}: {e}")
                return None

        slices = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(read_metadata, self.file_paths)
            for res in results:
                if res:
                    slices.append(res)
                    
        if not slices:
            return None
            
        # 2. Determine Dimensions
        unique_slice_locs = sorted(list(set(s['sl'] for s in slices)))
        unique_times = sorted(list(set(s['at'] for s in slices)))
        
        nz = len(unique_slice_locs)
        nt = len(unique_times)
        
        # Fallback for time sorting
        if nt < 2 and len(slices) > nz:
            slices.sort(key=lambda x: x['in'])
            nt = len(slices) // nz
            unique_times = list(range(nt))
            
        rows = slices[0]['rows']
        cols = slices[0]['cols']
        
        # Initialize Volume
        volume = np.zeros((rows, cols, nz, nt), dtype=np.float32)
        
        # Map slice/time to index
        z_map = {loc: i for i, loc in enumerate(unique_slice_locs)}
        
        # Organize tasks for pixel reading
        # We need to map each file to (z_idx, t_idx)
        
        slices_by_z = defaultdict(list)
        for s in slices:
            slices_by_z[s['sl']].append(s)
            
        read_tasks = []
        
        for z_loc, z_slices in slices_by_z.items():
            # Sort by Time/Instance
            z_slices.sort(key=lambda x: (x['at'], x['in']))
            z_idx = z_map[z_loc]
            
            for t_idx, s in enumerate(z_slices):
                if t_idx < nt:
                    read_tasks.append({
                        'path': s['path'],
                        'z_idx': z_idx,
                        't_idx': t_idx,
                        'slope': s['slope'],
                        'intercept': s['intercept']
                    })

        # 3. Read Pixels in Parallel
        # Function to read and process single slice
        def process_slice(task):
            try:
                ds = pydicom.dcmread(task['path'])
                img = ds.pixel_array.astype(np.float32) * task['slope'] + task['intercept']
                return task['z_idx'], task['t_idx'], img
            except Exception as e:
                return None
                
        # Use ThreadPoolExecutor for IO bound pixel reading
        # (pydicom reads from disk)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_slice, task) for task in read_tasks]
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                if res:
                    z, t, img = res
                    volume[:, :, z, t] = img
                    
        self.pixel_array = volume
        self.dims = volume.shape
        
        # Metadata dt
        if nt > 1 and len(unique_times) > 1 and max(unique_times) > 0:
             diffs = np.diff(unique_times)
             diffs = diffs[diffs > 0.1]
             if len(diffs) > 0:
                 self.metadata['dt'] = np.median(diffs)
                 
        if 'dt' not in self.metadata:
            self.metadata['dt'] = 1.0
            
        return volume

def scan_directory(path):
    """
    Scans a directory for DICOM files and groups them by Series.
    Returns a list of DicomSeries objects.
    """
    series_dict = defaultdict(list)
    
    # 1. Quick Scan
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.lower().endswith('.dcm') or '.' not in f:
                full_path = os.path.join(root, f)
                # We need SeriesUID.
                # Reading even stop_before_pixels might be slow if done sequentially for 1000s files.
                # But we need to group them.
                # Let's try to read lazily or parallelize this too?
                # Parallelizing the scan might be overkill if os.walk is fast, but dcmread is the bottleneck.
                # Let's just collect paths first.
                series_dict['all_paths'].append(full_path) # Temp storage

    # 2. Parallel Header Read to Group
    # Actually, we don't know the UID until we read it.
    # So we must read header.
    all_paths = []
    for root, dirs, files in os.walk(path):
        for f in files:
             if f.lower().endswith('.dcm') or '.' not in f:
                 all_paths.append(os.path.join(root, f))
                 
    # Map paths to UIDs
    path_uid_map = defaultdict(list)
    
    def get_uid(p):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True)
            return getattr(ds, 'SeriesInstanceUID', 'Unknown'), p
        except:
            return None

    if not all_paths:
        return []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = executor.map(get_uid, all_paths)
        for res in results:
            if res:
                uid, p = res
                path_uid_map[uid].append(p)
                
    # Create Series Objects
    result = []
    for uid, paths in path_uid_map.items():
        if uid != 'Unknown':
            result.append(DicomSeries(uid, paths))
            
    return result
