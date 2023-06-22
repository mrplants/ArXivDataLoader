"""
This script generates a Hugging Face Dataset from ArXiv research papers.
The research papers need to be downloaded separately using the ArXiv bulk data download.
"""

import csv
import json
import os
import tarfile
import gzip
import io
import logging
import re

import datasets

def sort_by_date(filenames):
    """Sorts a list of tar filenames by date"""
    # Use a regular expression to parse the dates
    pattern = re.compile(r'arXiv_src_(\d{2})(\d{2})_(\d{3})\.tar')

    def get_date(filename):
        """Extracts a date tuple from a filename"""
        match = pattern.match(filename)
        if match is None:
            return 0, 0, 0  # if the filename does not match, return a dummy value
        year, month, number = match.groups()
        year = int(year)
        if year < 90:  # if the year is less than 90, it is in the 2000s
            year += 2000
        else:  # if the year is 21 or more, it is 1900s
            year += 1900
        return year, int(month), int(number)

    return sorted(filenames, key=get_date)

# Create a custom logger
logger = logging.getLogger(__name__)

# Configure logger
logging.basicConfig(filename='arxiv_dataset.log', format='%(asctime)s:%(levelname)s:%(message)s', level=logging.INFO)

# Citation for ArXiv
_CITATION = """Thank you to arXiv for use of its open access interoperability."""

# Description of the Dataset
_DESCRIPTION = """This dataset enables training on ArXiv research papers, which must be downloaded separately using the ArXiv bulk data download."""

# The URL for the homepage of the dataset
_HOMEPAGE = "https://github.com/mrplants/arxiv_dataset"

# License of the ArXiv dataset
_LICENSE = "https://info.arxiv.org/help/license/index.html"

# The directory where the dataset is stored, it should be set as an environment variable
_URLS = [os.environ["ARXIV_DATASET_PATH"]]


class ArXivDataset(datasets.GeneratorBasedBuilder):
    """ A custom Hugging Face Dataset class for ArXiv research papers.
    """

    # Version of the dataset
    VERSION = datasets.Version("0.0.2")

    def _info(self):
        """ Returns dataset information such as features, homepage, license, and citation.
        """
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "id": datasets.Value("string"),
                "content": datasets.Value("string")
            }),
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """ Defines how to split the dataset.  There are no splits in this version.
        """
        dataset_url = _URLS[0]
        return [
            datasets.SplitGenerator(
                name="default",
                gen_kwargs={
                    "filepath": dataset_url,
                },
            )
        ]

    def _generate_examples(self, filepath):
        """ Generates dataset examples by iterating over tar files in the filepath and extracting relevant data.
        """
        key = 0
        for tar in reversed(sort_by_date(os.listdir(filepath))):
            # Skip non-tar files and hidden files
            if not tar.endswith(".tar") or tar.startswith("."):
                logging.info(f"Skipped non-tar or hidden file: {tar}")
                continue

            with tarfile.open(os.path.join(filepath, tar), 'r') as outer_tar:
                # Extract the gz file into a BytesIO stream
                ordered_members = sorted([f for f in outer_tar.getmembers() if f.name.endswith('.gz')], key=lambda f: f.name)
                for member in ordered_members:
                    gz_file_data = outer_tar.extractfile(member).read()
                    gz_file_stream = io.BytesIO(gz_file_data)

                    try:
                        # Open the stream as a tarfile
                        with tarfile.open(fileobj=gz_file_stream, mode='r:gz') as inner_tar:
                            # Iterate over the items in the tar file
                            for inner_member in inner_tar.getmembers():
                                # Only process 'main.tex' files (not directories or other files)
                                if inner_member.isfile() and inner_member.name == 'main.tex':
                                    # Extract file data
                                    file_data = inner_tar.extractfile(inner_member).read()

                                    # Yield the id (filename without extension) and content of the tex file
                                    yield key, {
                                        "id": inner_member.name[:-3],
                                        "content": file_data.decode('utf-8')
                                    }
                                    key += 1
                    except Exception as e:
                        # If there's an error reading a gz file, we skip it.
                        logging.error(f"Error processing file: {tar}/{member.name}. Error: {str(e)}")