from lsst.daf.butler.formatters.matplotlib import MatplotlibFormatter

__all__ = ["PdfMatplotlibFormatter"]


class PdfMatplotlibFormatter(MatplotlibFormatter):
    """Save figures in PDF format instead of PNG.

    A variant of `MatplotlibFormatter`, which Butler uses by default.
    """

    default_extension = ".pdf"
