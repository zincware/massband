import lazy_loader as lazy
import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# attach stdout logger
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
