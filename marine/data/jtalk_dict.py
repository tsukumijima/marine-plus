import asyncio
import json
import traceback
from pathlib import Path


try:
    import httpx
except ImportError:
    raise ImportError("httpx is required. Please install it with: pip install httpx")

DICTIONARIES_DIR = Path(__file__).parent / "dictionaries"
GITHUB_API_BASE = "https://api.github.com"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com"
REPO_NAME = "Aivis-Project/AivisSpeech-Engine"
DICT_PATH = "resources/dictionaries"
BRANCH = "master"
SHA_FILE = DICTIONARIES_DIR / "sha_info.json"


class DictionaryDownloader:
    """Dictionary downloader for OpenJTalk."""

    def __init__(self) -> None:
        """
        Initialize the dictionary downloader.
        """
        DICTIONARIES_DIR.mkdir(parents=True, exist_ok=True)
        self.sha_info: dict[str, str] = self._load_sha_info()

    def _load_sha_info(self) -> dict[str, str]:
        """
        Load SHA information from the JSON file.

        Returns:
            Dictionary containing filename to SHA mapping.
        """
        if SHA_FILE.exists():
            try:
                return json.loads(SHA_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                print("Warning: Invalid SHA info file. Starting fresh.")
                return {}
        return {}

    def _save_sha_info(self) -> None:
        """Save SHA information to the JSON file."""
        SHA_FILE.write_text(
            json.dumps(self.sha_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    async def _get_dict_files(self, client: httpx.AsyncClient) -> list[dict[str, str]]:
        """
        Get dictionary files information from GitHub API.

        Args:
            client: Async HTTP client.

        Returns:
            List of dictionary file information.
        """
        url = f"{GITHUB_API_BASE}/repos/{REPO_NAME}/contents/{DICT_PATH}"
        response = await client.get(url)
        response.raise_for_status()

        return sorted(
            [
                item
                for item in response.json()
                if item["type"] == "file" and item["name"].endswith(".dic")
            ],
            key=lambda x: x["name"],
        )

    async def _download_file(
        self, client: httpx.AsyncClient, file_info: dict[str, str]
    ) -> None:
        """
        Download a single dictionary file.

        Args:
            client: Async HTTP client.
            file_info: File information from GitHub API.
        """
        local_path = DICTIONARIES_DIR / file_info["name"]

        # Download file
        raw_url = (
            f'{GITHUB_RAW_BASE}/{REPO_NAME}/{BRANCH}/{DICT_PATH}/{file_info["name"]}'
        )
        response = await client.get(raw_url)
        response.raise_for_status()

        # Save file and update SHA info
        local_path.write_bytes(response.content)
        self.sha_info[file_info["name"]] = file_info["sha"]
        print(f'Downloaded: {file_info["name"]}')

    def _cleanup_obsolete_files(self, current_files: set[str]) -> None:
        """
        Remove dictionary files that no longer exist in the repository.

        Args:
            current_files: Set of filenames that exist in the repository.
        """
        for file in DICTIONARIES_DIR.glob("*.dic"):
            if file.name not in current_files:
                try:
                    file.unlink()
                    print(f"Removed obsolete file: {file.name}")
                except Exception:
                    print(f"Error removing {file.name}")
                    traceback.print_exc()
                if file.name in self.sha_info:
                    del self.sha_info[file.name]

    async def check_and_download_dictionaries(self) -> None:
        """Check for updates and download dictionaries if needed."""
        async with httpx.AsyncClient() as client:
            try:
                # Get dictionary files information from GitHub
                dict_files = await self._get_dict_files(client)
                download_tasks = []
                current_files = set()

                for file_info in dict_files:
                    current_files.add(file_info["name"])
                    local_path = DICTIONARIES_DIR / file_info["name"]
                    needs_download = True

                    if local_path.exists():
                        stored_sha = self.sha_info.get(file_info["name"])
                        if stored_sha == file_info["sha"]:
                            print(f'Skipping {file_info["name"]}: Already up to date.')
                            needs_download = False

                    if needs_download:
                        download_tasks.append(self._download_file(client, file_info))

                if download_tasks:
                    await asyncio.gather(*download_tasks)
                    self._save_sha_info()

                # Cleanup obsolete files
                self._cleanup_obsolete_files(current_files)

                if not download_tasks:
                    print("All dictionaries are up to date.")

            except httpx.HTTPError:
                print("Error downloading dictionaries:")
                traceback.print_exc()
            except Exception:
                print("Unexpected error:")
                traceback.print_exc()


def download_and_apply_dictionaries() -> None:
    """
    Download latest dictionaries if needed and apply them to pyopenjtalk.
    """
    try:
        import pyopenjtalk
    except ImportError:
        raise ImportError(
            'Please install pyopenjtalk by `pip install -e ".[dev,pyopenjtalk]"`'
        )

    # Download dictionaries if needed
    downloader = DictionaryDownloader()
    asyncio.run(downloader.check_and_download_dictionaries())

    # Get all .dic files in alphabetical order
    dict_files = sorted(DICTIONARIES_DIR.glob("*.dic"))
    if not dict_files:
        print("No dictionary files found.")
        return

    # Convert Path objects to resolved string paths
    dict_paths = [str(p.resolve(strict=True)) for p in dict_files]

    # Apply dictionaries to pyopenjtalk
    pyopenjtalk.update_global_jtalk_with_user_dict(dict_paths)
    print(f"Applied {len(dict_paths)} dictionaries to pyopenjtalk.")
