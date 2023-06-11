import xml.etree.ElementTree as ET
import os
from torch.utils.data import Dataset
import functools
import tarfile

class ArXivDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        with open(os.path.join(self.data_dir, 'src/arXiv_src_manifest.xml'), 'r') as f:
            self.manifest = self.parse_manifest_xml(f.read())
            self.__len__.cache_clear()

    @functools.lru_cache()
    def __len__(self):
        # Return the total number of papers.
        return sum([file['num_items'] for file in self.manifest])

    def __getitem__(self, idx):
        # Load and return the paper corresponding to idx,
        # as well as any associated metadata.
        tarfilename, subfile_idx = self.get_tar_for_index(idx)
        with tarfile.open(os.path.join(self.data_dir, tarfilename), 'r') as tar:
            ordered_members = sorted([f for f in tar.getmembers() if f.name.endswith('.gz')], key=lambda f: f.name)
            return ordered_members[subfile_idx].name
    
    def get_tar_for_index(self, idx):
        current_index = 0
        for file_data in self.manifest:
            if current_index + file_data['num_items'] > idx:
                return file_data['filename'], idx - current_index
            current_index += file_data['num_items']
        
        return None, None

    def parse_manifest_xml(self, xml_string):
        # Parse the XML string
        root = ET.fromstring(xml_string)
        
        files_data = []
        # Iterate over all 'file' elements
        for file in root.findall('file'):
            file_data = {
                'filename': file.find('filename').text,
                'num_items': int(file.find('num_items').text),
                'seq_num': int(file.find('seq_num').text),
                'yymm': file.find('yymm').text
            }
            file_data['year'] = int('20'+file_data['yymm'][:2] if int(file_data['yymm'][:2]) <= 23 else '19'+file_data['yymm'][:2])
            file_data['month'] = int(file_data['yymm'][2:])

            files_data.append(file_data)
        
        files_data = sorted(files_data, key=lambda x: (x['year'], x['month'], x['seq_num']))
        
        return files_data