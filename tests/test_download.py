# tests/test_download.py
import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from aioresponses import aioresponses

from shard.download import (
    DownloadStatus,
    DownloadTask,
    DownloadStats,
    DownloadManager
)


class TestDownloadStatus:
    """Test suite for DownloadStatus enum"""

    def test_download_status_values(self):
        """Test DownloadStatus enum values"""
        assert DownloadStatus.PENDING.value == "pending"
        assert DownloadStatus.DOWNLOADING.value == "downloading"
        assert DownloadStatus.COMPLETED.value == "completed"
        assert DownloadStatus.FAILED.value == "failed"

    def test_download_status_unique(self):
        """Test all DownloadStatus values are unique"""
        values = [status.value for status in DownloadStatus]
        assert len(values) == len(set(values))


class TestDownloadTask:
    """Test suite for DownloadTask dataclass"""

    def test_download_task_creation(self, tmp_path):
        """Test DownloadTask creation"""
        lock = asyncio.Lock()
        task = DownloadTask(
            uri="https://example.com/file.bin",
            partition="test-partition",
            path=tmp_path / "file.bin",
            total_size=1024,
            downloaded=512,
            status=DownloadStatus.DOWNLOADING,
            claims=2,
            lock=lock
        )

        assert task.uri == "https://example.com/file.bin"
        assert task.partition == "test-partition"
        assert task.path == tmp_path / "file.bin"
        assert task.total_size == 1024
        assert task.downloaded == 512
        assert task.status == DownloadStatus.DOWNLOADING
        assert task.claims == 2
        assert task.lock == lock


class TestDownloadStats:
    """Test suite for DownloadStats dataclass"""

    def test_download_stats_creation(self):
        """Test DownloadStats creation"""
        stats = DownloadStats(
            active_workers=2,
            completed_jobs=5,
            failed_jobs=1,
            total_downloaded=1024,
            total_size=2048
        )

        assert stats.active_workers == 2
        assert stats.completed_jobs == 5
        assert stats.failed_jobs == 1
        assert stats.total_downloaded == 1024
        assert stats.total_size == 2048

    def test_progress_pct_calculation(self):
        """Test progress_pct property calculates correctly"""
        stats = DownloadStats(
            active_workers=1,
            completed_jobs=0,
            failed_jobs=0,
            total_downloaded=512,
            total_size=1024
        )
        assert stats.progress_pct == 50.0

    def test_progress_pct_zero_total_size(self):
        """Test progress_pct returns 0 when total_size is 0"""
        stats = DownloadStats(
            active_workers=0,
            completed_jobs=0,
            failed_jobs=0,
            total_downloaded=0,
            total_size=0
        )
        assert stats.progress_pct == 0.0

    def test_progress_pct_complete(self):
        """Test progress_pct when download is complete"""
        stats = DownloadStats(
            active_workers=0,
            completed_jobs=1,
            failed_jobs=0,
            total_downloaded=1024,
            total_size=1024
        )
        assert stats.progress_pct == 100.0


class TestDownloadManager:
    """Test suite for DownloadManager"""

    def test_download_manager_initialization(self, tmp_path):
        """Test DownloadManager initialization"""
        manager = DownloadManager(
            storage_path=tmp_path,
            progress_interval=2.0,
            clean_cache=True,
            http_timeout=5000.0
        )

        assert manager.storage_path == tmp_path
        assert manager.progress_interval == 2.0
        assert manager.clean_cache is True
        assert manager.http_timeout == 5000.0
        assert len(manager.downloads) == 0
        assert len(manager.progress_callbacks) == 0

    def test_clean_filename_simple(self, tmp_path):
        """Test _clean_filename with simple URL"""
        manager = DownloadManager(storage_path=tmp_path)
        filename = manager._clean_filename("https://example.com/path/file.bin")
        assert filename == "file.bin"

    def test_clean_filename_with_query_params(self, tmp_path):
        """Test _clean_filename strips query parameters"""
        manager = DownloadManager(storage_path=tmp_path)
        filename = manager._clean_filename("https://example.com/file.bin?download=true&token=abc")
        assert filename == "file.bin"

    def test_clean_filename_with_encoded_chars(self, tmp_path):
        """Test _clean_filename handles URL-encoded characters"""
        manager = DownloadManager(storage_path=tmp_path)
        filename = manager._clean_filename("https://example.com/my%20file.bin")
        assert filename == "my file.bin"

    def test_get_partition_path_creates_directory(self, tmp_path):
        """Test _get_partition_path creates directory if not exists"""
        manager = DownloadManager(storage_path=tmp_path)
        partition_path = manager._get_partition_path("test-partition")

        assert partition_path == tmp_path / "test-partition"
        assert partition_path.exists()
        assert partition_path.is_dir()

    def test_get_partition_path_existing_directory(self, tmp_path):
        """Test _get_partition_path with existing directory"""
        partition_dir = tmp_path / "existing-partition"
        partition_dir.mkdir()

        manager = DownloadManager(storage_path=tmp_path)
        partition_path = manager._get_partition_path("existing-partition")

        assert partition_path == partition_dir
        assert partition_path.exists()

    async def test_cache_file_new_download(self, tmp_path):
        """Test cache_file creates new download task"""
        manager = DownloadManager(storage_path=tmp_path)

        result = await manager.cache_file(
            partition="test-partition",
            uri="https://example.com/file.bin",
            no_claims=3
        )

        assert result is False  # New download
        download_key = ("test-partition", "https://example.com/file.bin")
        assert download_key in manager.downloads

        task = manager.downloads[download_key]
        assert task.uri == "https://example.com/file.bin"
        assert task.partition == "test-partition"
        assert task.claims == 3
        assert task.status == DownloadStatus.PENDING

    async def test_cache_file_existing_download(self, tmp_path):
        """Test cache_file returns True for existing download"""
        manager = DownloadManager(storage_path=tmp_path)

        # First call
        await manager.cache_file(
            partition="test-partition",
            uri="https://example.com/file.bin",
            no_claims=2
        )

        # Second call
        result = await manager.cache_file(
            partition="test-partition",
            uri="https://example.com/file.bin",
            no_claims=3
        )

        assert result is True
        download_key = ("test-partition", "https://example.com/file.bin")
        task = manager.downloads[download_key]
        assert task.claims == 3  # Updated claims

    async def test_cache_file_existing_file(self, tmp_path):
        """Test cache_file with already downloaded file"""
        # Create a pre-existing file
        partition_dir = tmp_path / "test-partition"
        partition_dir.mkdir()
        test_file = partition_dir / "file.bin"
        test_file.write_bytes(b"test data")

        manager = DownloadManager(storage_path=tmp_path)

        result = await manager.cache_file(
            partition="test-partition",
            uri="https://example.com/file.bin",
            no_claims=2
        )

        assert result is True
        download_key = ("test-partition", "https://example.com/file.bin")
        task = manager.downloads[download_key]
        assert task.status == DownloadStatus.COMPLETED
        assert task.total_size == len(b"test data")
        assert task.downloaded == task.total_size

    async def test_get_file_not_registered(self, tmp_path):
        """Test get_file raises RuntimeError for unregistered download"""
        manager = DownloadManager(storage_path=tmp_path)

        with pytest.raises(RuntimeError, match="No download registered"):
            await manager.get_file("test-partition", "https://example.com/file.bin")

    async def test_get_file_completed(self, tmp_path):
        """Test get_file returns path for completed download"""
        # Setup completed download
        partition_dir = tmp_path / "test-partition"
        partition_dir.mkdir()
        test_file = partition_dir / "file.bin"
        test_file.write_bytes(b"test data")

        manager = DownloadManager(storage_path=tmp_path)
        await manager.cache_file(
            partition="test-partition",
            uri="https://example.com/file.bin",
            no_claims=2
        )

        path = await manager.get_file("test-partition", "https://example.com/file.bin", claim=True)

        assert path == test_file
        download_key = ("test-partition", "https://example.com/file.bin")
        assert manager.downloads[download_key].claims == 1  # Decremented

    async def test_get_file_failed(self, tmp_path):
        """Test get_file raises RuntimeError for failed download"""
        manager = DownloadManager(storage_path=tmp_path)

        # Create a failed download task
        download_key = ("test-partition", "https://example.com/file.bin")
        manager.downloads[download_key] = DownloadTask(
            uri="https://example.com/file.bin",
            partition="test-partition",
            path=tmp_path / "test-partition" / "file.bin",
            total_size=0,
            downloaded=0,
            status=DownloadStatus.FAILED,
            claims=1,
            lock=asyncio.Lock()
        )

        with pytest.raises(RuntimeError, match="Failed to download"):
            await manager.get_file("test-partition", "https://example.com/file.bin")

    async def test_get_file_clean_cache_zero_claims(self, tmp_path):
        """Test get_file removes file when clean_cache=True and claims=0"""
        # Setup completed download
        partition_dir = tmp_path / "test-partition"
        partition_dir.mkdir()
        test_file = partition_dir / "file.bin"
        test_file.write_bytes(b"test data")

        manager = DownloadManager(storage_path=tmp_path, clean_cache=True)
        await manager.cache_file(
            partition="test-partition",
            uri="https://example.com/file.bin",
            no_claims=1
        )

        # First claim
        path = await manager.get_file("test-partition", "https://example.com/file.bin", claim=True)
        assert path == test_file

        # Second call with zero claims should remove file
        path = await manager.get_file("test-partition", "https://example.com/file.bin", claim=False)
        assert path is None
        download_key = ("test-partition", "https://example.com/file.bin")
        assert download_key not in manager.downloads

    async def test_get_file_no_claim(self, tmp_path):
        """Test get_file with claim=False doesn't decrement claims"""
        partition_dir = tmp_path / "test-partition"
        partition_dir.mkdir()
        test_file = partition_dir / "file.bin"
        test_file.write_bytes(b"test data")

        manager = DownloadManager(storage_path=tmp_path)
        await manager.cache_file(
            partition="test-partition",
            uri="https://example.com/file.bin",
            no_claims=2
        )

        path = await manager.get_file("test-partition", "https://example.com/file.bin", claim=False)

        assert path == test_file
        download_key = ("test-partition", "https://example.com/file.bin")
        assert manager.downloads[download_key].claims == 2  # Not decremented

    async def test_add_progress_callback_valid(self, tmp_path):
        """Test add_progress_callback with valid async function"""
        manager = DownloadManager(storage_path=tmp_path)

        async def async_callback(stats):
            pass

        manager.add_progress_callback(async_callback)
        assert len(manager.progress_callbacks) == 1

    async def test_add_progress_callback_invalid(self, tmp_path):
        """Test add_progress_callback raises error for non-async function"""
        manager = DownloadManager(storage_path=tmp_path)

        def sync_callback(stats):
            pass

        with pytest.raises(ValueError, match="Callback must be a coroutine function"):
            manager.add_progress_callback(sync_callback)

    async def test_cleanup_all_partitions(self, tmp_path):
        """Test cleanup removes all files when partition=None"""
        # Setup multiple partitions with files
        for partition in ["partition1", "partition2"]:
            partition_dir = tmp_path / partition
            partition_dir.mkdir()
            test_file = partition_dir / "file.bin"
            test_file.write_bytes(b"test data")

        manager = DownloadManager(storage_path=tmp_path)

        # Register downloads
        for partition in ["partition1", "partition2"]:
            await manager.cache_file(partition, f"https://example.com/{partition}/file.bin")

        # Cleanup all
        await manager.cleanup()

        assert len(manager.downloads) == 0
        assert not (tmp_path / "partition1" / "file.bin").exists()
        assert not (tmp_path / "partition2" / "file.bin").exists()

    async def test_cleanup_specific_partition(self, tmp_path):
        """Test cleanup removes files only from specific partition"""
        # Setup multiple partitions
        for partition in ["partition1", "partition2"]:
            partition_dir = tmp_path / partition
            partition_dir.mkdir()
            test_file = partition_dir / "file.bin"
            test_file.write_bytes(b"test data")

        manager = DownloadManager(storage_path=tmp_path)

        # Register downloads
        for partition in ["partition1", "partition2"]:
            await manager.cache_file(partition, f"https://example.com/{partition}/file.bin")

        # Cleanup only partition1
        await manager.cleanup(partition="partition1")

        assert len(manager.downloads) == 1
        assert not (tmp_path / "partition1" / "file.bin").exists()
        assert (tmp_path / "partition2" / "file.bin").exists()

    async def test_download_file_success(self, tmp_path):
        """Test _download_file successfully downloads a file"""
        manager = DownloadManager(storage_path=tmp_path)

        # Setup download task
        download_key = ("test-partition", "https://example.com/file.bin")
        dest_path = tmp_path / "test-partition" / "file.bin"
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        task = DownloadTask(
            uri="https://example.com/file.bin",
            partition="test-partition",
            path=dest_path,
            total_size=0,
            downloaded=0,
            status=DownloadStatus.PENDING,
            claims=1,
            lock=asyncio.Lock()
        )
        manager.downloads[download_key] = task

        # Mock the HTTP response
        test_data = b"test file content"
        with aioresponses() as mocked:
            mocked.get(
                "https://example.com/file.bin",
                status=200,
                body=test_data,
                headers={"content-length": str(len(test_data))}
            )

            # _download_file expects the lock to be acquired before calling
            await task.lock.acquire()
            await manager._download_file(download_key)

        assert task.status == DownloadStatus.COMPLETED
        assert task.total_size == len(test_data)
        assert task.downloaded == len(test_data)
        assert dest_path.exists()
        assert dest_path.read_bytes() == test_data

    async def test_download_file_http_error(self, tmp_path):
        """Test _download_file handles HTTP errors"""
        manager = DownloadManager(storage_path=tmp_path)

        download_key = ("test-partition", "https://example.com/file.bin")
        dest_path = tmp_path / "test-partition" / "file.bin"
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        task = DownloadTask(
            uri="https://example.com/file.bin",
            partition="test-partition",
            path=dest_path,
            total_size=0,
            downloaded=0,
            status=DownloadStatus.PENDING,
            claims=1,
            lock=asyncio.Lock()
        )
        manager.downloads[download_key] = task

        # Mock HTTP 404 error
        with aioresponses() as mocked:
            mocked.get("https://example.com/file.bin", status=404)

            # _download_file expects the lock to be acquired before calling
            await task.lock.acquire()
            await manager._download_file(download_key)

        assert task.status == DownloadStatus.FAILED
        assert not dest_path.exists()

    async def test_download_file_size_mismatch(self, tmp_path):
        """Test _download_file handles size mismatch error"""
        manager = DownloadManager(storage_path=tmp_path)

        download_key = ("test-partition", "https://example.com/file.bin")
        dest_path = tmp_path / "test-partition" / "file.bin"
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        task = DownloadTask(
            uri="https://example.com/file.bin",
            partition="test-partition",
            path=dest_path,
            total_size=0,
            downloaded=0,
            status=DownloadStatus.PENDING,
            claims=1,
            lock=asyncio.Lock()
        )
        manager.downloads[download_key] = task

        # Mock response with mismatched content-length
        test_data = b"short"
        with aioresponses() as mocked:
            mocked.get(
                "https://example.com/file.bin",
                status=200,
                body=test_data,
                headers={"content-length": "1000"}  # Mismatch
            )

            await task.lock.acquire()
            await manager._download_file(download_key)

        assert task.status == DownloadStatus.FAILED

    async def test_download_file_creates_temp_file(self, tmp_path):
        """Test _download_file uses temp file during download"""
        manager = DownloadManager(storage_path=tmp_path)

        download_key = ("test-partition", "https://example.com/file.bin")
        dest_path = tmp_path / "test-partition" / "file.bin"
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        task = DownloadTask(
            uri="https://example.com/file.bin",
            partition="test-partition",
            path=dest_path,
            total_size=0,
            downloaded=0,
            status=DownloadStatus.PENDING,
            claims=1,
            lock=asyncio.Lock()
        )
        manager.downloads[download_key] = task

        test_data = b"test content"
        with aioresponses() as mocked:
            mocked.get(
                "https://example.com/file.bin",
                status=200,
                body=test_data,
                headers={"content-length": str(len(test_data))}
            )

            await task.lock.acquire()
            await manager._download_file(download_key)

        # Temp file should be removed
        temp_path = dest_path.with_suffix('.tmp')
        assert not temp_path.exists()
        assert dest_path.exists()

    async def test_get_file_pending(self, tmp_path):
        """Test get_file returns None for pending download"""
        manager = DownloadManager(storage_path=tmp_path)

        download_key = ("test-partition", "https://example.com/file.bin")
        manager.downloads[download_key] = DownloadTask(
            uri="https://example.com/file.bin",
            partition="test-partition",
            path=tmp_path / "test-partition" / "file.bin",
            total_size=0,
            downloaded=0,
            status=DownloadStatus.PENDING,
            claims=1,
            lock=asyncio.Lock()
        )

        path = await manager.get_file("test-partition", "https://example.com/file.bin", claim=False)
        assert path is None
