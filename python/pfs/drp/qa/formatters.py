from lsst.daf.butler.formatters.matplotlib import MatplotlibFormatter


__all__ = ["PdfMatplotlibFormatter"]


class PdfMatplotlibFormatter(MatplotlibFormatter):
    """Variant of `MatplotlibFormatter` (used by Butler) that saves figures
    in PDF format instead of PNG format.
    """

    extension = ".pdf"
