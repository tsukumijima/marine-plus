import asyncio
import json
import tempfile
from pathlib import Path
from typing import Final

from marine.logger import getLogger


try:
    import httpx
except ImportError:
    raise ImportError("httpx is required. Please install it with: pip install httpx")

DICTIONARIES_DIR = Path(__file__).parent / "dictionaries"
SHA_FILE = DICTIONARIES_DIR / "sha_info.json"

logger = getLogger()


class DictionaryDownloader:
    """
    OpenJTalk 用デフォルトユーザー辞書のダウンローダー。

    デフォルト辞書の更新を確認し、必要であればダウンロードする。
    """

    GITHUB_API_BASE: Final[str] = "https://api.github.com"
    GITHUB_RAW_BASE: Final[str] = "https://raw.githubusercontent.com"
    REPO_NAME: Final[str] = "Aivis-Project/AivisSpeech-Engine"
    DICT_PATH: Final[str] = "resources/dictionaries"
    BRANCH: Final[str] = "master"

    def __init__(self) -> None:
        """
        Initialize the dictionary downloader.
        """
        DICTIONARIES_DIR.mkdir(parents=True, exist_ok=True)
        self.sha_info: dict[str, str] = self._load_sha_info()

    def _load_sha_info(self) -> dict[str, str]:
        """
        SHA 情報を JSON ファイルから読み込む。

        Returns:
            ファイル名と SHA のマッピングを含む辞書。
        """
        if SHA_FILE.exists():
            try:
                return json.loads(SHA_FILE.read_text(encoding="utf-8"))
            except json.JSONDecodeError as ex:
                logger.warning("Invalid SHA info file. Starting fresh.", exc_info=ex)
                return {}
        return {}

    def _save_sha_info(self) -> None:
        """SHA 情報を JSON ファイルに保存する。"""
        SHA_FILE.write_text(
            json.dumps(self.sha_info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    async def _get_dict_files(self) -> list[dict[str, str]]:
        """
        GitHub API から辞書ファイル情報を取得する。

        Returns:
            辞書ファイル情報のリスト。
        """
        async with httpx.AsyncClient() as client:
            url = f"{self.GITHUB_API_BASE}/repos/{self.REPO_NAME}/contents/{self.DICT_PATH}"
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

    async def _download_file(self, file_info: dict[str, str]) -> tuple[str, str]:
        """
        単一の辞書ファイルをダウンロードする。
        成功した場合は (ファイル名, SHA) のタプルを返す。

        Args:
            file_info: GitHub API から取得したファイル情報。
        """
        local_path = DICTIONARIES_DIR / file_info["name"]
        temp_file_path: Path | None = None
        file_name = file_info["name"]  # ログ出力用にファイル名を変数に

        async with httpx.AsyncClient() as client:
            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, dir=DICTIONARIES_DIR, suffix=".tmp"
                ) as tmpfile:
                    temp_file_path = Path(tmpfile.name)

                raw_url = f"{self.GITHUB_RAW_BASE}/{self.REPO_NAME}/{self.BRANCH}/{self.DICT_PATH}/{file_name}"
                logger.info(f"[DictionaryDownloader][{file_name}] Downloading...")
                response = await client.get(raw_url)
                response.raise_for_status()

                temp_file_path.write_bytes(response.content)
                temp_file_path.rename(local_path)
                logger.info(f"[DictionaryDownloader][{file_name}] Download complete.")
                return file_info["name"], file_info[
                    "sha"
                ]  # 成功時にファイル名と SHA を返す

            except Exception as ex:
                logger.error(
                    f"[DictionaryDownloader][{file_name}] Error during download process",
                    exc_info=ex,
                )
                raise ex

    def _cleanup_obsolete_files(self, current_files: set[str]) -> None:
        """
        リポジトリに存在しなくなった辞書ファイルを削除する。

        Args:
            current_files: リポジトリに存在するファイル名のセット。
        """
        for file in DICTIONARIES_DIR.glob("*.dic"):
            if file.name not in current_files:
                try:
                    file.unlink()
                    logger.info(
                        f"[DictionaryDownloader] Removed obsolete file: {file.name}"
                    )
                except Exception as ex:
                    logger.error(
                        f"[DictionaryDownloader] Error removing {file.name}",
                        exc_info=ex,
                    )
                if file.name in self.sha_info:
                    del self.sha_info[file.name]

    async def check_and_download_dictionaries(self) -> bool:
        """
        更新を確認し、必要であれば辞書をダウンロードする。

        Returns:
            bool: 辞書のダウンロードが発生した場合は True、そうでない場合は False。
        """
        download_occurred = False
        try:
            dict_files = await self._get_dict_files()
            logger.info(
                f"[DictionaryDownloader] Found {len(dict_files)} dictionary files on remote."
            )
            download_tasks = []
            current_files = set()
            queued_file_infos: list[
                dict[str, str]
            ] = []  # ダウンロードキューに入れられたファイル情報を保持

            for file_info in dict_files:
                current_files.add(file_info["name"])
                local_path = DICTIONARIES_DIR / file_info["name"]
                needs_download = True

                if local_path.exists():
                    stored_sha = self.sha_info.get(file_info["name"])
                    if stored_sha == file_info["sha"]:
                        logger.info(
                            f"[DictionaryDownloader] [{file_info['name']}] Skipping: Already up to date."
                        )
                        needs_download = False
                    else:
                        logger.info(
                            f"[DictionaryDownloader] [{file_info['name']}] SHA mismatch. Stored: {stored_sha}, Remote: {file_info['sha']}. Needs download."
                        )

                if needs_download:
                    download_tasks.append(self._download_file(file_info))
                    queued_file_infos.append(file_info)  # 対応する file_info を保存

            if download_tasks:
                results = await asyncio.gather(*download_tasks, return_exceptions=True)

                successful_downloads_count = 0
                any_sha_updated_in_run = False

                for i, result in enumerate(results):
                    task_file_info = queued_file_infos[i]
                    log_file_name = task_file_info["name"]

                    if isinstance(result, Exception):
                        logger.error(
                            f"[DictionaryDownloader][{log_file_name}] Download task failed in gather: {result!r}"
                        )
                        # このファイルの SHA 情報は更新しない
                    elif isinstance(result, tuple) and len(result) == 2:
                        # 成功時は (file_name, sha) タプルが返ってくる
                        downloaded_file_name, downloaded_sha = result
                        if (
                            downloaded_file_name == log_file_name
                        ):  # 念のためファイル名を確認
                            self.sha_info[downloaded_file_name] = downloaded_sha
                            logger.info(
                                f"[DictionaryDownloader][{log_file_name}] Successfully downloaded. SHA info updated to {downloaded_sha[:8]}."
                            )
                            successful_downloads_count += 1
                            any_sha_updated_in_run = (
                                True  # 少なくとも1つの SHA が更新された
                            )
                        else:
                            # これは予期せぬ状況
                            logger.error(
                                f"[DictionaryDownloader][{log_file_name}] Mismatch in downloaded file name. Expected {log_file_name}, got {downloaded_file_name}. SHA not updated."
                            )
                    else:
                        # 予期せぬ result の型 (Exception でも tuple でもない場合)
                        logger.error(
                            f"[DictionaryDownloader][{log_file_name}] Unknown result type from download task: {type(result)}. SHA not updated."
                        )

                logger.info(
                    f"[DictionaryDownloader] {successful_downloads_count} out of {len(download_tasks)} download tasks completed successfully."
                )

                if any_sha_updated_in_run:
                    logger.info(
                        "[DictionaryDownloader] At least one file was successfully downloaded and SHA updated. Saving SHA info."
                    )
                    self._save_sha_info()
                    download_occurred = True  # 実際に変更があったので True
                else:
                    logger.info(
                        "[DictionaryDownloader] No files were successfully downloaded or updated in this run. SHA info will not be saved."
                    )
                    # download_occurred は False のまま
            else:
                logger.info(
                    "[DictionaryDownloader] No download tasks were queued. All files seem up to date based on initial check."
                )

            self._cleanup_obsolete_files(current_files)

            if not download_tasks:
                logger.info(
                    "[DictionaryDownloader] All dictionaries are up to date (no tasks queued)."
                )
            elif download_occurred:
                logger.info(
                    "[DictionaryDownloader] Download process finished with updates."
                )
            else:
                logger.warning(
                    "[DictionaryDownloader] Download tasks were queued, but no files were successfully updated."
                )

            return download_occurred

        except httpx.HTTPError as ex:
            logger.error(
                "[DictionaryDownloader] Error during dictionary download (HTTPError):",
                exc_info=ex,
            )
            return False
        except Exception as ex:
            logger.error(
                "[DictionaryDownloader] Unexpected error during dictionary download process:",
                exc_info=ex,
            )
            return False


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
        logger.warning("[DictionaryDownloader] No dictionary files found.")
        return

    # Convert Path objects to resolved string paths
    dict_paths = [str(p.resolve(strict=True)) for p in dict_files]

    # Apply dictionaries to pyopenjtalk
    pyopenjtalk.update_global_jtalk_with_user_dict(dict_paths)
    logger.info(
        f"[DictionaryDownloader] Applied {len(dict_paths)} dictionaries to pyopenjtalk."
    )
