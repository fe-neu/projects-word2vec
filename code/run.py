from pathlib import Path
import time

class Run:
    def __init__(
        self,
        base_path: Path = Path("runs"),
        run_name: str | None = None,
        description: str = "",
    ):
        self.base_path = base_path
        self.run_name = run_name or f"run_{int(time.time())}"
        self.run_path = self.base_path / self.run_name
        self.description = description

        self._create_run_dir()
        self._write_description()

    def _create_run_dir(self):
        self.run_path.mkdir(parents=True, exist_ok=False)

    def _write_description(self):
        if not self.description:
            return

        desc_path = self.run_path / "description.txt"
        desc_path.write_text(self.description, encoding="utf-8")
