"""bedtools intersect — three common overlap queries against a peaks file.

This example consumes ``modules/bedtools_intersect.py`` (a typed shell wrapper
around the ``bedtools intersect`` CLI) and exercises three of its most-used
flag combinations on a small BED fixture:

- ``wa=True`` — write each A feature that has *any* overlap in B.
- ``v=True``  — write each A feature that has *no* overlap in B (set diff).
- ``c=True``  — write each A feature with a trailing count of B overlaps.

Fixture (4 "genes" in A, 3 "peaks" in B, all on chr1):

    A (genes)               B (peaks)
    chr1 100-200 gene1      chr1 150-180 peak1   <- overlaps gene1
    chr1 300-400 gene2      chr1 350-450 peak2   <- overlaps gene2
    chr1 500-600 gene3      chr1 900-950 peak3
    chr1 700-800 gene4

Expected:
- wa  -> gene1, gene2
- v   -> gene3, gene4
- c   -> gene1\\t1, gene2\\t1, gene3\\t0, gene4\\t0

Run locally::

    uv run python 12_bedtools_intersect_example.py
"""

import asyncio
import tempfile

import flyte
from flyte.io import File

from modules.bedtools_intersect import bedtools_intersect


env = flyte.TaskEnvironment(
    name="bedtools_intersect_example",
    depends_on=[bedtools_intersect.env],
)


@env.task
async def intersect_demo(genes: File, peaks: File) -> tuple[File, File, File]:
    overlapping, non_overlapping, counts = await asyncio.gather(
        bedtools_intersect(a=genes, b=[peaks], wa=True),
        bedtools_intersect(a=genes, b=[peaks], v=True),
        bedtools_intersect(a=genes, b=[peaks], count_overlaps=True),
    )
    return overlapping, non_overlapping, counts


GENES_BED = (
    "chr1\t100\t200\tgene1\t0\t+\n"
    "chr1\t300\t400\tgene2\t0\t+\n"
    "chr1\t500\t600\tgene3\t0\t+\n"
    "chr1\t700\t800\tgene4\t0\t+\n"
)

PEAKS_BED = (
    "chr1\t150\t180\tpeak1\t0\t+\n"
    "chr1\t350\t450\tpeak2\t0\t+\n"
    "chr1\t900\t950\tpeak3\t0\t+\n"
)




# Fixtures
# mkdir -p /tmp/bedtools-fixtures && \
# printf 'chr1\t100\t200\tgene1\t0\t+\nchr1\t300\t400\tgene2\t0\t+\nchr1\t500\t600\tgene3\t0\t+\nchr1\t700\t800\tgene4\t0\t+\n' > /tmp/bedtools-fixtures/genes.bed && \
# printf 'chr1\t150\t180\tpeak1\t0\t+\nchr1\t350\t450\tpeak2\t0\t+\nchr1\t900\t950\tpeak3\t0\t+\n' > /tmp/bedtools-fixtures/peaks.bed && \
# ls -la /tmp/bedtools-fixtures/

if __name__ == "__main__":

    flyte.init()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write(GENES_BED)
        genes_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".bed", delete=False) as f:
        f.write(PEAKS_BED)
        peaks_path = f.name

    run = flyte.with_runcontext().run(
        intersect_demo,
        File.from_local_sync(genes_path),
        File.from_local_sync(peaks_path),
    )

    print(run)

    out = run.outputs()
    overlapping_path = out.o0.download_sync("./overlapping.bed")
    non_overlapping_path = out.o1.download_sync("./non_overlapping.bed")
    counts_path = out.o2.download_sync("./counts.bed")
    print(f"Wrote: {overlapping_path}, {non_overlapping_path}, {counts_path}")
