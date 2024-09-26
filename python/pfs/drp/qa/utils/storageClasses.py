import io
from typing import Union

from lsst.daf.butler import StorageClass
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

__all__ = [
    "MultipagePdfFigure",
    "QaDict",
]


class MultipagePdfFigure(StorageClass):
    """Duck type for `matplotlib.figure.Figure`,
    to be passed to ``butler.put()``.

    This is a wrapper of `matplotlib.backends.backend_pdf.PdfPages`.
    The behavior of ``MultipagePdfFigure.savefig()`` conforms to that of
    ``Figure.savefig()``.
    The counterpart of ``PdfPages.savefig()`` is ``MultipagePdfFigure.append()``.

    Parameters
    ----------
    *args
        Arguments passed to the constructor of `PdfPages`.
    **kwargs
        Arguments passed to the constructor of `PdfPages`.
    """

    def __init__(self, *args, **kwargs) -> None:
        self._memfile = io.BytesIO()
        self.pdfpages = PdfPages(self._memfile, *args, **kwargs)

    def savefig(self, fname: str, *args, **kwargs) -> None:
        """Save this PDF to a file.

        Parameters
        ----------
        fname : `str`
            File name.
        *args
            Ignored.
        **kwargs
            Ignored.
        """
        self.pdfpages.close()
        content = self._memfile.getvalue()
        if not content:
            # We follow `PdfPages`, which does not (over)write anything
            # if no page has been made.
            return
        with open(fname, "wb") as f:
            f.write(content)

    def append(self, figure: Union[Figure, int, None] = None, **kwargs) -> None:
        """Append a page.

        Parameters
        ----------
        figure : `Union[Figure, int, None]`
            Figure drawn on the newly added page.
        **kwargs:
            Passed to `PdfPages.savefig()`.
        """
        self.pdfpages.savefig(figure, **kwargs)


class QaDict(dict):
    """Dict type for QA results."""

    pass
