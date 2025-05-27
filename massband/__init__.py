import lazy_loader as lazy
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# attach stdout logger


__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
