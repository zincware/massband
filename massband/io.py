import contextlib
import typing

import h5py
import znh5md
import zntrack


class LoadIOFile(zntrack.Node):
    path: str = zntrack.deps_path()

    def run(self):
        pass

    @property
    def frames(self) -> znh5md.IO:
        @contextlib.contextmanager
        def _factory() -> typing.Callable[[], typing.ContextManager[h5py.File]]:
            if self.state.rev is None and self.state.remote is None:
                with h5py.File(self.path, "r") as file:
                    yield file
            else:
                with self.state.fs.open(self.path, "rb") as f:
                    with h5py.File(f) as file:
                        yield file

        return znh5md.IO(file_factory=_factory)
