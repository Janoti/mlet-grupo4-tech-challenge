from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Caminhos resolvidos utilizados pelo projeto."""

    root: Path
    src: Path
    data: Path
    docs: Path
    notebooks: Path
    models: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "ProjectPaths":
        project_root = root or Path(__file__).resolve().parents[3]

        return cls(
            root=project_root,
            src=project_root / "src",
            data=project_root / "data",
            docs=project_root / "docs",
            notebooks=project_root / "notebooks",
            models=project_root / "models",
        )


paths = ProjectPaths.from_root()