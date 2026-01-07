from pathlib import Path
import time


class Run:
    """
    Utility class for managing experiment run directories.

    A run consists of a uniquely named directory under a base path, optionally
    containing a textual description of the experiment. This class is intended
    to provide lightweight run tracking for experiments and training runs.
    """

    def __init__(
        self,
        base_path: Path = Path("runs"),
        run_name: str | None = None,
        description: str = "",
    ):
        """
        Initialize a new run and create its directory structure.

        If no run name is provided, a timestamp-based name is generated
        automatically. The run directory is created immediately, and an
        optional description is written to disk.

        :param base_path: Base directory under which all runs are stored.
        :type base_path: Path
        :param run_name: Optional name for the run directory. If ``None``,
                         a timestamp-based name is generated.
        :type run_name: str | None
        :param description: Optional textual description of the run.
        :type description: str
        """
        self.base_path = base_path
        self.run_name = run_name or f"run_{int(time.time())}"
        self.run_path = self.base_path / self.run_name
        self.description = description

        self._create_run_dir()
        self._write_description()

    def _create_run_dir(self):
        """
        Create the directory for the current run.

        The directory is created recursively. If the directory already exists,
        an exception is raised.
        """
        self.run_path.mkdir(parents=True, exist_ok=False)

    def _write_description(self):
        """
        Write the run description to a text file.

        If no description is provided, this method performs no action. When
        present, the description is written to ``description.txt`` inside the
        run directory using UTF-8 encoding.
        """
        if not self.description:
            return

        desc_path = self.run_path / "description.txt"
        desc_path.write_text(self.description, encoding="utf-8")
