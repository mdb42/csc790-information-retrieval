"""
data_loader.py - Module for downloading and managing datasets
"""

import os
import logging
import zipfile
import urllib.request
import shutil
import hashlib

class DataLoader:
    """
    Manages dataset downloads and verification for the search engine.
    Handles downloading, extracting, and validating public datasets.
    """
    
    def __init__(self, data_dir="data/datasets"):
        """
        Initialize the data loader.
        
        Args:
            data_dir (str): Directory to store downloaded datasets
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger('dataloader')
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Dataset definitions with metadata
        self.datasets = {
            "goodreads_reviews": {
                "name": "Goodreads Book Reviews",
                "url": "https://github.com/user/goodreads-datasets/releases/download/v1.0/goodreads_reviews.zip",
                "filename": "goodreads_reviews.zip",
                "md5": "abcdef1234567890abcdef1234567890",  # Replace with actual MD5
                "size_mb": 150,
                "extract_dir": "goodreads_reviews",
                "description": "10M book reviews from Goodreads"
            },
            "gutenberg_books": {
                "name": "Project Gutenberg Selection",
                "url": "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2",
                "filename": "rdf-files.tar.bz2",
                "extract_dir": "gutenberg",
                "description": "Metadata for Project Gutenberg books"
            },
            # Add more datasets as needed
        }
    
    def list_available_datasets(self):
        """
        List all available datasets with their status (downloaded or not).
        
        Returns:
            list: List of dataset information dictionaries
        """
        result = []
        for key, dataset in self.datasets.items():
            dataset_path = os.path.join(self.data_dir, dataset["filename"])
            extracted_path = os.path.join(self.data_dir, dataset["extract_dir"])
            
            result.append({
                "id": key,
                "name": dataset["name"],
                "description": dataset["description"],
                "downloaded": os.path.exists(dataset_path),
                "extracted": os.path.exists(extracted_path),
                "size_mb": dataset.get("size_mb", "Unknown")
            })
        
        return result
    
    def download_dataset(self, dataset_id, force=False):
        """
        Download a specific dataset.
        
        Args:
            dataset_id (str): ID of the dataset to download
            force (bool): Whether to force re-download if already exists
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dataset_id not in self.datasets:
            self.logger.error(f"Unknown dataset: {dataset_id}")
            return False
        
        dataset = self.datasets[dataset_id]
        file_path = os.path.join(self.data_dir, dataset["filename"])
        
        # Skip if already downloaded unless force is True
        if os.path.exists(file_path) and not force:
            self.logger.info(f"Dataset {dataset['name']} already downloaded")
            return True
        
        # Download the file
        self.logger.info(f"Downloading {dataset['name']}...")
        try:
            urllib.request.urlretrieve(
                dataset["url"], 
                file_path,
                reporthook=self._download_progress_hook(dataset)
            )
            
            # Verify checksum if provided
            if "md5" in dataset:
                if not self._verify_md5(file_path, dataset["md5"]):
                    self.logger.error(f"MD5 verification failed for {dataset['name']}")
                    os.remove(file_path)
                    return False
            
            self.logger.info(f"Successfully downloaded {dataset['name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading {dataset['name']}: {str(e)}")
            # Clean up partial download
            if os.path.exists(file_path):
                os.remove(file_path)
            return False
    
    def extract_dataset(self, dataset_id, force=False):
        """
        Extract a downloaded dataset.
        
        Args:
            dataset_id (str): ID of the dataset to extract
            force (bool): Whether to force re-extraction if already extracted
            
        Returns:
            bool: True if successful, False otherwise
        """
        if dataset_id not in self.datasets:
            self.logger.error(f"Unknown dataset: {dataset_id}")
            return False
            
        dataset = self.datasets[dataset_id]
        file_path = os.path.join(self.data_dir, dataset["filename"])
        extract_dir = os.path.join(self.data_dir, dataset["extract_dir"])
        
        # Check if downloaded
        if not os.path.exists(file_path):
            self.logger.error(f"Dataset {dataset['name']} not downloaded yet")
            return False
            
        # Skip if already extracted unless force is True
        if os.path.exists(extract_dir) and not force:
            self.logger.info(f"Dataset {dataset['name']} already extracted")
            return True
            
        # Create extraction directory
        os.makedirs(extract_dir, exist_ok=True)
        
        # Extract based on file type
        try:
            self.logger.info(f"Extracting {dataset['name']}...")
            
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            elif file_path.endswith('.tar.gz') or file_path.endswith('.tgz'):
                import tarfile
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(path=extract_dir)
            elif file_path.endswith('.tar.bz2'):
                import tarfile
                with tarfile.open(file_path, 'r:bz2') as tar:
                    tar.extractall(path=extract_dir)
            else:
                self.logger.error(f"Unsupported archive format for {dataset['name']}")
                return False
                
            self.logger.info(f"Successfully extracted {dataset['name']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error extracting {dataset['name']}: {str(e)}")
            # Clean up partial extraction
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            return False
    
    def get_dataset_path(self, dataset_id):
        """
        Get the path to an extracted dataset.
        
        Args:
            dataset_id (str): ID of the dataset
            
        Returns:
            str: Path to the extracted dataset directory or None if not available
        """
        if dataset_id not in self.datasets:
            return None
            
        extract_dir = os.path.join(self.data_dir, self.datasets[dataset_id]["extract_dir"])
        
        if os.path.exists(extract_dir):
            return extract_dir
        return None
    
    def ensure_dataset(self, dataset_id):
        """
        Ensure a dataset is downloaded and extracted, downloading if necessary.
        
        Args:
            dataset_id (str): ID of the dataset
            
        Returns:
            str: Path to the extracted dataset or None if failed
        """
        # Download if needed
        if not self.download_dataset(dataset_id):
            return None
            
        # Extract if needed
        if not self.extract_dataset(dataset_id):
            return None
            
        return self.get_dataset_path(dataset_id)
    
    def _verify_md5(self, file_path, expected_md5):
        """
        Verify the MD5 hash of a file.
        
        Args:
            file_path (str): Path to the file
            expected_md5 (str): Expected MD5 hash
            
        Returns:
            bool: True if the MD5 matches, False otherwise
        """
        self.logger.info(f"Verifying MD5 for {file_path}...")
        
        md5_hash = hashlib.md5()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                md5_hash.update(chunk)
                
        file_md5 = md5_hash.hexdigest()
        if file_md5 != expected_md5:
            self.logger.error(f"MD5 mismatch: expected {expected_md5}, got {file_md5}")
            return False
            
        return True
    
    def _download_progress_hook(self, dataset):
        """
        Create a progress hook for downloads.
        
        Args:
            dataset (dict): Dataset information
            
        Returns:
            function: Progress hook function for urlretrieve
        """
        def hook(count, block_size, total_size):
            if total_size > 0:
                percent = min(int(count * block_size * 100 / total_size), 100)
                if percent % 10 == 0:
                    self.logger.info(f"Downloading {dataset['name']}: {percent}% complete")
        return hook