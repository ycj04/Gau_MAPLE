"""Gaussian External process invocation parsing.

Gaussian appends six positional arguments to the command supplied with the
``External`` keyword::

    layer InputFile OutputFile MsgFile FChkFile MatElFile

Gau_MAPLE treats those as the final six arguments so model-selection flags may
precede them in the Gaussian route section.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .errors import InvocationError


@dataclass(frozen=True, slots=True)
class GaussianInvocation:
    layer: str
    input_path: Path
    output_path: Path
    message_path: Path
    formatted_checkpoint_path: Path
    matrix_element_path: Path
    option_argv: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        layer = str(self.layer).strip().upper()
        if layer not in {"R", "M", "S"}:
            raise InvocationError(
                f"Unknown Gaussian External layer {self.layer!r}; expected R, M, or S."
            )
        object.__setattr__(self, "layer", layer)
        for field_name in (
            "input_path",
            "output_path",
            "message_path",
            "formatted_checkpoint_path",
            "matrix_element_path",
        ):
            value = Path(getattr(self, field_name)).expanduser()
            object.__setattr__(self, field_name, value)
        object.__setattr__(
            self,
            "option_argv",
            tuple(str(item) for item in self.option_argv),
        )

    def validate_direct_mode(self) -> "GaussianInvocation":
        """Validate the subset supported by the direct molecular adapter."""
        if self.layer != "R":
            raise InvocationError(
                "Gau_MAPLE direct mode currently supports only Gaussian's real-system "
                "layer 'R'. ONIOM model/small layers M and S are not implemented."
            )
        if not self.input_path.is_file():
            raise InvocationError(
                f"Gaussian External input file does not exist: {self.input_path}"
            )

        output_resolved = self.output_path.absolute()
        message_resolved = self.message_path.absolute()
        input_resolved = self.input_path.absolute()
        if output_resolved == message_resolved:
            raise InvocationError("OutputFile and MsgFile must be different paths.")
        if output_resolved == input_resolved:
            raise InvocationError("InputFile and OutputFile must be different paths.")
        return self


def parse_gaussian_invocation(argv: Sequence[str]) -> GaussianInvocation:
    """Parse command arguments, taking the final six as Gaussian-owned fields."""
    values = [str(item) for item in argv]
    if len(values) < 6:
        raise InvocationError(
            "Gaussian External supplies six final arguments: layer, InputFile, "
            "OutputFile, MsgFile, FChkFile, and MatElFile. "
            f"Received only {len(values)} argument(s)."
        )
    option_argv = tuple(values[:-6])
    layer, input_file, output_file, msg_file, fchk_file, matel_file = values[-6:]
    return GaussianInvocation(
        layer=layer,
        input_path=Path(input_file),
        output_path=Path(output_file),
        message_path=Path(msg_file),
        formatted_checkpoint_path=Path(fchk_file),
        matrix_element_path=Path(matel_file),
        option_argv=option_argv,
    )
