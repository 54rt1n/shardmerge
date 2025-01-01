# shard/download.py
# Copyright (C) 2024 Martin Bukowski
#
# This software is free software: you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program. If not, see http://www.gnu.org/licenses/.

from dataclasses import dataclass
from typing import Dict, Optional, Callable, List, Tuple
import asyncio
import os
import aiohttp
import logging
from pathlib import Path
import time
from enum import Enum
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)

class DownloadStatus(Enum):
    """Status states for download tracking"""
    PENDING = "pending"
    DOWNLOADING = "downloading"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class DownloadTask:
    """
    Represents a download task in the system
    
    Attributes:
        uri: Source URL for the download
        partition: Storage partition identifier (typically model name)
        path: Local filesystem path for the downloaded file
        total_size: Expected size of the download in bytes
        downloaded: Current bytes downloaded
        status: Current DownloadStatus of this task
        claims: Number of active claims on this download
    """
    uri: str
    partition: str
    path: Path
    total_size: int
    downloaded: int
    status: DownloadStatus
    claims: int
    lock: asyncio.Lock

@dataclass
class DownloadStats:
    """
    System-wide download statistics
    
    Attributes:
        active_workers: Number of currently downloading tasks
        completed_jobs: Number of successfully completed downloads
        failed_jobs: Number of failed downloads
        total_downloaded: Total bytes downloaded across all tasks
        total_size: Total expected bytes across all tasks
    """
    active_workers: int
    completed_jobs: int
    failed_jobs: int
    total_downloaded: int
    total_size: int

    @property
    def progress_pct(self) -> float:
        """Calculate overall download progress as percentage"""
        if self.total_size == 0:
            return 0.0
        return (self.total_downloaded / self.total_size) * 100

class DownloadManager:
    """
    Manages multiple concurrent downloads with partitioned storage
    
    This class handles the orchestration of multiple download tasks,
    manages storage partitioning, and provides progress tracking.
    """
    
    def __init__(
        self, 
        storage_path: Path,
        progress_interval: float = 1.0,
        clean_cache: bool = False,
        http_timeout: float = 3600.0,
    ):
        """
        Initialize the download manager
        
        Args:
            storage_path: Root directory for all downloads
            progress_interval: Minimum seconds between progress updates
            clean_cache: Whether to delete cached files on claims
            http_timeout: HTTP timeout for downloads
        """
        self.storage_path = storage_path
        self.progress_interval = progress_interval
        self.last_progress_time = 0.0
        self.clean_cache = clean_cache
        self.http_timeout = http_timeout
        
        # Track downloads by (partition, uri)
        self.downloads: Dict[Tuple[str, str], DownloadTask] = {}
        self.progress_callbacks: List[Callable[[DownloadStats], None]] = []

    def _clean_filename(self, uri: str) -> str:
        """
        Clean a URI to get a safe filename
        
        Args:
            uri: The URI to clean
            
        Returns:
            str: Cleaned filename without query parameters
        """
        parsed = urlparse(uri)
        # Get the path part and remove query parameters
        filename = os.path.basename(unquote(parsed.path))
        return filename

    def _get_partition_path(self, partition: str) -> Path:
        """Get the path for a partition's storage directory"""
        partition_path = self.storage_path / partition
        partition_path.mkdir(parents=True, exist_ok=True)
        return partition_path

    async def cache_file(self, partition: str, uri: str, no_claims: int = 1) -> bool:
        """
        Register a file for downloading in the specified partition
        
        Args:
            partition: Storage partition identifier
            uri: Source URL to download
            no_claims: Number of initial claims on this download
            
        Returns:
            bool: True if file exists or is being downloaded, False if new download started
        """
        download_key = (partition, uri)
        if download_key in self.downloads:
            task = self.downloads[download_key]
            task.claims = no_claims
            return True

        # Create new download task
        clean_filename = self._clean_filename(uri)
        dest_path = self._get_partition_path(partition) / clean_filename
        task = DownloadTask(
            uri=uri,
            partition=partition,
            path=dest_path,
            total_size=0,
            downloaded=0,
            status=DownloadStatus.PENDING,
            claims=no_claims,
            lock=asyncio.Lock(),
        )
        self.downloads[download_key] = task

        if dest_path.exists():
            task.status = DownloadStatus.COMPLETED
            task.total_size = dest_path.stat().st_size
            task.downloaded = task.total_size
            await self._check_progress(force=True)
            return True

        # Start download task
        logger.info(f"Starting download of {uri} to {dest_path}")
        await task.lock.acquire()
        asyncio.create_task(self._download_file(download_key))
        return False

    async def get_file(
        self, 
        partition: str, 
        uri: str, 
        claim: bool = True, 
    ) -> Optional[Path]:
        """
        Get the path to a downloaded file in the specified partition
        
        Args:
            partition: Storage partition identifier
            uri: Source URL of the file
            claim: Whether to decrement claims counter
            cleanup: Whether to remove file if claims reach zero
            
        Returns:
            Optional[Path]: Path to downloaded file if available
            
        Raises:
            RuntimeError: If the download failed
        """
        file_key = (partition, uri)
        if file_key not in self.downloads:
            raise RuntimeError(f"No download registered for {uri}")
            
        file = self.downloads[file_key]

        if self.clean_cache and file.claims <= 0:
            logger.debug(f"Removing {file.path} due to zero claims")
            if file.path.exists():
                logger.debug(f"Removing {file.path}")
                file.path.unlink()
            del self.downloads[file_key]
            return None
        
        if claim:
            file.claims -= 1
            
        if file.status == DownloadStatus.FAILED:
            raise RuntimeError(f"Failed to download {uri}")
            
        if file.status == DownloadStatus.COMPLETED:
            return file.path
            
        return None

    async def _download_file(self, download_key: Tuple[str, str]):
        """Handle the actual file download"""
        task = self.downloads[download_key]
        temp_path = task.path.with_suffix('.tmp')
        
        try:
            task.status = DownloadStatus.DOWNLOADING
            await self._check_progress(force=True)
            
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.http_timeout)
            ) as session:
                async with session.get(task.uri) as response:
                    response.raise_for_status()
                    
                    # Get content length, default to 0 if unknown
                    content_length = response.headers.get('content-length')
                    task.total_size = int(content_length) if content_length else 0
                    logger.debug(f"Download size for {task.uri}: {task.total_size} bytes")
                    
                    with open(temp_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                                task.downloaded += len(chunk)
                                await self._check_progress()

            # Check if the file was downloaded successfully
            if temp_path.stat().st_size != task.total_size:
                logger.error(f"Downloaded file size does not match content-length header for {task.uri}")
                raise RuntimeError(f"Downloaded file size does not match content-length header for {task.uri}")

            # Move temp file to final location
            import shutil
            task_path = shutil.move(temp_path, task.path)
            logger.info(f"Moved {temp_path} to {task_path} ({task.path})")
            
            # Check if the file exists
            if not task.path.exists():
                raise RuntimeError(f"Downloaded file disappeared: {task.path} {task_path}")

            import glob
            for filename in glob.glob(str(task.path) + ".*"):
                logger.info(f"Found {filename}")
            
            task.status = DownloadStatus.COMPLETED
            logger.info(f"Download completed for {task.uri}, written to {task.path}")
            
        except Exception as e:
            task.status = DownloadStatus.FAILED
            logger.error(f"Download failed for {task.uri}: {str(e)}")
            import traceback
            traceback.print_exc()
            if temp_path.exists():
                logger.info(f"Removing {temp_path}")
                temp_path.unlink()
        
        finally:
            await self._check_progress(force=True)
            # Release the lock
            task.lock.release()

    def add_progress_callback(self, callback: Callable[[DownloadStats], None]):
        """Add a callback to be called on progress updates"""
        # We need to make sure our callback is async, so we check the type and throw an error if it's not
        if not asyncio.iscoroutinefunction(callback):
            raise ValueError("Callback must be a coroutine function")
        self.progress_callbacks.append(callback)

    async def _check_progress(self, force: bool = False):
        """
        Check if we should send a progress update
        
        Args:
            force: Whether to force an update regardless of time interval
        """
        current_time = time.time()
        if not force and (current_time - self.last_progress_time < self.progress_interval):
            return
            
        self.last_progress_time = current_time
        
        # Calculate current statistics
        active = sum(1 for task in self.downloads.values() 
                    if task.status == DownloadStatus.DOWNLOADING)
        completed = sum(1 for task in self.downloads.values() 
                       if task.status == DownloadStatus.COMPLETED)
        failed = sum(1 for task in self.downloads.values() 
                    if task.status == DownloadStatus.FAILED)
                    
        total_downloaded = sum(task.downloaded for task in self.downloads.values() if task.status == DownloadStatus.DOWNLOADING)
        total_size = sum(task.total_size for task in self.downloads.values() if task.status == DownloadStatus.DOWNLOADING)

        stats = DownloadStats(
            active_workers=active,
            completed_jobs=completed,
            failed_jobs=failed,
            total_downloaded=total_downloaded,
            total_size=total_size
        )

        # Log current stats
        logger.debug(
            f"Download stats - Active: {active}, Completed: {completed}, "
            f"Failed: {failed}, Progress: {stats.progress_pct:.1f}%, "
            f"Downloaded: {total_downloaded/1024:.1f}KB"
        )

        for callback in self.progress_callbacks:
            await callback(stats)

    async def cleanup(self, partition: Optional[str] = None):
        """
        Clean up downloaded files
        
        Args:
            partition: Optional partition to clean. If None, cleans all partitions
        """
        keys_to_remove = []
        for (part, uri), task in self.downloads.items():
            if partition is None or part == partition:
                if task.path.exists():
                    logger.info(f"Removing cache file: {task.path}")
                    task.path.unlink()
                keys_to_remove.append((part, uri))
                
        for key in keys_to_remove:
            del self.downloads[key]
