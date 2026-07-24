"""Tests for parsing nsys stats CSV and rendering the report HTML (pure, no nsys binary)."""

import flyteplugins.nsight._report as rep

# Representative nsys stats CSV (columns track nsys 2023+; values are illustrative).
KERN_CSV = (
    "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
    "70.0,7000000,100,70000,70000,60000,80000,5000,ampere_sgemm_128x64\n"
    "30.0,3000000,50,60000,60000,50000,70000,4000,elementwise_kernel\n"
)

MEM_SIZE_CSV = (
    "Total (MB),Count,Avg (MB),Med (MB),Min (MB),Max (MB),StdDev (MB),Operation\n"
    "512.0,20,25.6,25.6,25.6,25.6,0.0,[CUDA memcpy HtoD]\n"
    "128.0,10,12.8,12.8,12.8,12.8,0.0,[CUDA memcpy DtoH]\n"
)

NVTX_CSV = (
    "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Range\n"
    "60.0,6000000,20,300000,300000,250000,350000,20000,step\n"
    "40.0,4000000,20,200000,200000,180000,220000,10000,forward\n"
)

# An unquoted comma in a demangled kernel name makes the data row longer than the header, so
# csv.DictReader stashes the overflow under a None restkey (regression: crashed _col's key.lower()).
KERN_OVERFLOW_CSV = (
    "Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name\n"
    "100.0,7000000,100,70000,70000,60000,80000,5000,void kernel<int, float>\n"
)

# nsys stats --format csv prints status lines to stdout before the CSV. Parsing must skip them,
# else the first status line becomes the header and every real column reads as empty (regression:
# report tiles/bars all showed 0 / "(unnamed)").
KERN_WITH_PREAMBLE = (
    "Generating SQLite file /tmp/nsys/a0/report.sqlite from /tmp/nsys/a0/report.nsys-rep\n"
    "Processing [/tmp/nsys/a0/report.sqlite] with [/opt/nsys/reports/cuda_gpu_kern_sum.py]...\n"
    "\n"
    " ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):\n"
    "\n"
) + KERN_CSV


class TestParsing:
    def test_rows_parses_csv(self):
        rows = rep._rows(KERN_CSV)
        assert len(rows) == 2
        assert rows[0]["Name"] == "ampere_sgemm_128x64"

    def test_rows_empty_on_none(self):
        assert rep._rows(None) == []
        assert rep._rows("") == []

    def test_col_fuzzy_match(self):
        row = rep._rows(KERN_CSV)[0]
        assert rep._col(row, "Total Time") == "7000000"
        assert rep._col(row, "Name") == "ampere_sgemm_128x64"

    def test_col_time_percent_not_confused_with_total_time(self):
        # The bug guard: "Time (%)" must not resolve to "Total Time (ns)".
        row = rep._rows(KERN_CSV)[0]
        assert rep._col(row, "Time", "%") == "70.0"

    def test_rows_drops_overflow_restkey(self):
        # A row longer than the header must not leave a None key behind, and the numeric columns
        # before the overflow stay intact so the summary still renders.
        rows = rep._rows(KERN_OVERFLOW_CSV)
        assert None not in rows[0]
        assert rep._col(rows[0], "Total Time") == "7000000"
        parsed = {"cuda_gpu_kern_sum": rows, "cuda_gpu_mem_size_sum": [], "nvtx_pushpop_sum": []}
        _html, summary = rep._summary_tiles(parsed)
        assert summary["gpu_kernel_time_ns"] == 7000000

    def test_col_tolerates_none_key(self):
        # Defensive: even if a None key reaches _col directly, it must not crash.
        assert rep._col({None: ["x"], "Total Time (ns)": "42"}, "Total Time") == "42"

    def test_rows_strips_nsys_preamble(self):
        # The status lines nsys prints before the CSV must be skipped so real columns resolve.
        rows = rep._rows(KERN_WITH_PREAMBLE)
        assert len(rows) == 2
        assert rows[0]["Name"] == "ampere_sgemm_128x64"
        assert rep._col(rows[0], "Total Time") == "7000000"

    def test_rows_empty_when_no_csv(self):
        # An empty / "SKIPPED" report has no comma line: no rows, no crash.
        assert rep._rows("SKIPPED: report does not apply to this trace\n") == []

    def test_fmt_ns_units(self):
        assert rep._fmt_ns(7_000_000) == "7.00 ms"
        assert rep._fmt_ns(2_000_000_000) == "2.00 s"
        assert rep._fmt_ns(500) == "500 ns"


class TestShortName:
    def test_cutlass_extracts_inner_variant(self):
        n = "void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(T1::Params)"
        assert rep._short_name(n) == "cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4"

    def test_anonymous_namespace_kernel(self):
        n = (
            "void at::native::<unnamed>::multi_tensor_apply_kernel<at::native::<unnamed>::"
            "TensorListMetadata<(int)2>, std::plus<float>, float>(T1, T2, T3...)"
        )
        assert rep._short_name(n) == "multi_tensor_apply_kernel"

    def test_long_signature_collapses_to_identifier(self):
        n = "void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float>>(int, T2, T3)"
        assert rep._short_name(n) == "vectorized_elementwise_kernel"

    def test_nvtx_range_drops_leading_colon(self):
        assert rep._short_name(":forward") == "forward"
        assert rep._short_name(":step_0") == "step_0"

    def test_plain_name_passthrough(self):
        assert rep._short_name("Memset (Device)") == "Memset"
        assert rep._short_name("cudaLaunchKernel") == "cudaLaunchKernel"


_ELEM_MSE = (
    "void at::native::vectorized_elementwise_kernel<(int)4, at::native::mse_kernel_cuda"
    "(at::TensorIteratorBase &), at::detail::Array<char *, (int)3>>(int, T2, T3)"
)
_ELEM_FILL = (
    "void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, "
    "at::detail::Array<char *, (int)1>>(int, T2)"
)
_REDUCE_MEAN = (
    "void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, "
    "at::native::MeanOps<float, float, float, float>, unsigned int, float, (int)4>>(T3)"
)


class TestFunctorHint:
    def test_elementwise_functors_distinguished(self):
        assert rep._functor_hint(_ELEM_MSE) == "mse_kernel_cuda"
        assert rep._functor_hint(_ELEM_FILL) == "FillFunctor"

    def test_reduce_descends_through_wrapper(self):
        assert rep._functor_hint(_REDUCE_MEAN) == "MeanOps"

    def test_no_template_args_returns_empty(self):
        assert rep._functor_hint("cudaLaunchKernel") == ""
        assert rep._functor_hint("Memset (Device)") == ""


class TestSummaryTiles:
    def test_summary_aggregates(self):
        parsed = {
            "cuda_gpu_kern_sum": rep._rows(KERN_CSV),
            "cuda_gpu_mem_size_sum": rep._rows(MEM_SIZE_CSV),
            "nvtx_pushpop_sum": rep._rows(NVTX_CSV),
        }
        html, summary = rep._summary_tiles(parsed)
        assert summary["kernel_launches"] == 150
        assert summary["distinct_kernels"] == 2
        assert summary["top_kernel"] == "ampere_sgemm_128x64"
        assert summary["nvtx_ranges"] == 2
        # 10,000,000 ns total kernel time -> "10.00 ms"
        assert "10.00 ms" in html
        # memory tiles resolved by operation label
        assert "512.0 MB" in html
        assert "128.0 MB" in html

    def test_mem_unit_read_from_header_not_assumed(self):
        # The size unit comes from the column header ("Total (GB)"), not a hard-coded "MB".
        gb = "Total (GB),Count,Avg (GB),Operation\n2.500,4,0.625,[CUDA memcpy Host-to-Device]\n"
        parsed = {"cuda_gpu_kern_sum": [], "cuda_gpu_mem_size_sum": rep._rows(gb), "nvtx_pushpop_sum": []}
        html, _ = rep._summary_tiles(parsed)
        assert "2.5 GB" in html
        assert "MB" not in html


class TestRenderBody:
    def test_trace_url_linked_when_given(self):
        # Clustered workers pass a durable storage path; it must be linked from the deck.
        html, _ = rep._render_body(
            {"cuda_gpu_kern_sum": rep._rows(KERN_CSV)}, "GPU profile", trace_url="s3://bucket/run/report.nsys-rep"
        )
        assert "s3://bucket/run/report.nsys-rep" in html
        assert "flyte storage cp" in html

    def test_default_note_without_trace_url(self):
        html, _ = rep._render_body({"cuda_gpu_kern_sum": rep._rows(KERN_CSV)}, "GPU profile")
        assert "Download the .nsys-rep trace output" in html
        assert "flyte storage cp" not in html


class TestBars:
    def test_bars_ranks_by_total_time(self):
        html = rep._bars(rep._rows(KERN_CSV), ("Name",), "Top kernels")
        assert "ampere_sgemm_128x64" in html
        assert "Top kernels" in html
        # top row shows its percentage
        assert "70.0%" in html

    def test_bars_empty_on_no_rows(self):
        assert rep._bars([], ("Name",), "Top kernels") == ""

    def test_bars_disambiguate_colliding_kernels(self):
        rows = [
            {"Time (%)": "1.4", "Total Time (ns)": "1243999", "Name": _ELEM_MSE},
            {"Time (%)": "0.9", "Total Time (ns)": "792608", "Name": _ELEM_FILL},
        ]
        html = rep._bars(rows, ("Name",), "Top kernels")
        assert "vectorized_elementwise_kernel · mse_kernel_cuda" in html
        assert "vectorized_elementwise_kernel · FillFunctor" in html


class TestTable:
    def test_table_renders_rows(self):
        html = rep._table(rep._rows(KERN_CSV), "Kernel summary")
        assert "Kernel summary" in html
        assert "elementwise_kernel" in html
        assert "<table" in html

    def test_table_renders_all_rows_no_truncation(self):
        # Every row must render (the detail table is the expandable full view) — no "+N more" cap.
        csv = "Time (%),Total Time (ns),Name\n" + "".join(f"{i},{i}000,kernel_{i}\n" for i in range(30))
        html = rep._table(rep._rows(csv), "Big table")
        assert "(30 rows)" in html
        assert "kernel_29" in html  # last row present, not truncated
        assert "more rows" not in html
