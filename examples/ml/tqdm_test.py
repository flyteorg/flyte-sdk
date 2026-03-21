from tqdm import tqdm
import time

class HtmlTqdm(tqdm):
    def __init__(self, *args, html_path="progress.html", **kwargs):
        self.html_path = html_path
        # open file and write header
        self._html_file = open(self.html_path, "w")
        self._html_file.write("<html><body><h3>Progress</h3><pre>\n")
        super().__init__(*args, **kwargs)

    def display(self, msg=None, pos=None):
        """Override display to log HTML instead of printing to console."""
        if not self.disable:
            s = msg or self.__str__()
            self._html_file.write(s + "\n")
            self._html_file.flush()  # flush so progress updates immediately

    def close(self):
        """Make sure to close file properly."""
        super().close()
        self._html_file.write("</pre></body></html>\n")
        self._html_file.close()


def main():
    for i in HtmlTqdm(range(10), html_path="progress.html"):
        time.sleep(0.2)



if __name__ == "__main__":
    main()

