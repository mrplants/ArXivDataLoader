import xml.etree.ElementTree as ET
import os
import logging
from torch.utils.data import Dataset
import functools
import tarfile
from datetime import datetime

class ArXivDataset(Dataset):
    """
    A PyTorch Dataset for handling the Cornell's ArXiv research paper dataset.
    It can load and parse the XML manifest and provide access to the papers in 
    the dataset using a PyTorch DataLoader.
    
    Example Usage:
        dataset = ArXivDataset('/path/to/dataset')
        print(len(dataset))
        print(dataset[12345])  # This will give you the 12345th paper

    Attributes:
        data_dir (str): The directory where the dataset resides.
        manifest (list): A list of metadata for all files in the dataset.
        verbosity (int): Logging level. If set to 0, logging is disabled. 
    """
    def __init__(self, data_dir, verbosity=0, remove_abstracts=False):
        """
        Args:
            data_dir (str): The directory where the dataset resides.
            verbosity (int, optional): Logging level. If set to 0, logging is disabled. 
                                       Defaults to 0.
        """
        self.data_dir = data_dir
        self.remove_abstracts = remove_abstracts
        self.logger = logging.getLogger(__name__)
        if verbosity == 0:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

        try:
            with open(os.path.join(self.data_dir, 'src/arXiv_src_manifest.xml'), 'r') as f:
                self.manifest = self.parse_manifest_xml(f.read())
                self.__len__.cache_clear()
        except Exception as e:
            self.logger.error("Failed to open or parse manifest XML. Error: %s", e)

    @functools.lru_cache()
    def __len__(self):
        """Returns the total number of papers."""
        return sum([file['num_items'] for file in self.manifest])

    def __getitem__(self, idx):
        """
        Load and return the paper corresponding to idx,
        as well as any associated metadata.
        
        Args:
            idx (int): The index of the paper to be returned.
        
        Returns:
            str: The name of the corresponding paper.
        """
        # Handle backwards indexing
        if idx < 0:
            idx = len(self) + idx
        tarfilename, subfile_idx = self.get_tar_for_index(idx)
        try:
            with tarfile.open(os.path.join(self.data_dir, tarfilename), 'r') as tar:
                ordered_members = sorted([f for f in tar.getmembers() if f.name.endswith('.gz') or f.name.endswith('.pdf')], key=lambda f: f.name)
                return ordered_members[subfile_idx].name
        except FileNotFoundError:
            logging.error("Tar file not found.")
            raise

    def get_tar_for_index(self, idx):
        """
        Retrieves the tar file and the index within that tar file for a given global index.
        
        Args:
            idx (int): The global index of the paper.
        
        Returns:
            (str, int): A tuple containing the filename of the tar file and the index within that tar file.
        """
        current_index = 0
        for file_data in self.manifest:
            if current_index + file_data['num_items'] > idx:
                return file_data['filename'], idx - current_index
            current_index += file_data['num_items']
        
        raise IndexError(f'Index {idx} out of range')

    def parse_manifest_xml(self, xml_string):
        """
        Parse the XML string of the dataset manifest.
        
        Args:
            xml_string (str): XML string of the manifest.
        
        Returns:
            list: A list of dictionaries containing the metadata of each file in the manifest.
        """
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError:
            logging.error("Error while parsing XML.")
            raise
        
        files_data = []
        for file in root.findall('file'):
            file_data = {
                'filename': file.find('filename').text,
                'num_items': int(file.find('num_items').text),
                'seq_num': int(file.find('seq_num').text),
                'yymm': file.find('yymm').text
            }
            file_data['year'] = int('20'+file_data['yymm'][:2] if int(file_data['yymm'][:2]) <= datetime.now().year-2000 else '19'+file_data['yymm'][:2])
            file_data['month'] = int(file_data['yymm'][2:])
            files_data.append(file_data)
        
        files_data = sorted(files_data, key=lambda x: (x['year'], x['month'], x['seq_num']))
        
        return files_data