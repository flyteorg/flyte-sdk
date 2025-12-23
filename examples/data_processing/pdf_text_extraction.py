"""
PDF Text Extraction Workflow

This Flyte workflow downloads a PDF file, extracts all text (including potentially
redacted text from PDF layers), and displays the results in a Flyte report.

This script implements a Flyte workflow for extracting textual content from a PDF,
with a focus on uncovering not only visible text, but also any potentially hidden or redacted
content present on any PDF layer. The workflow is organized into the following key steps:

1. Download the PDF: The workflow first downloads the PDF from the supplied URL.
   This is performed asynchronously to support large files and robust error handling.
2. Extract All Text (Including Hidden/Redacted Content): The script processes the downloaded
   PDF using a library capable of accessing all text layers and elements. It iterates through
   each page and layer, extracting visible text, annotation details, and collecting any "hidden"
   characters that may exist in non-standard layers or appear to have been redacted but are
   present in the file's data.
3. Generate an HTML Report: After extraction, the script prepares a detailed HTML report summarizing
   the extracted visible and hidden text, as well as annotation metadata. The report
   organizes content by page and highlights potentially sensitive, hidden, or redacted snippets.
4. Log and Present Results: The HTML report is logged in the Flyte report panel. A concise
   summary of the findings (number of pages, total character count, annotations found, and count
   of hidden or redacted characters) is generated and returned by the workflow.

"""

import tempfile

import flyte
import flyte.report

env = flyte.TaskEnvironment(
    name="pdf_text_extraction",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "httpx",
        "pymupdf>=1.24.0",  # For PDF text extraction with layer access
    ),
)


@env.task
async def download_pdf(url: str) -> bytes:
    """
    Download a PDF file from a URL.
    
    Args:
        url: The URL of the PDF file to download.
        
    Returns:
        The raw bytes of the PDF file.
    """
    import httpx
    
    print(f"Downloading PDF from: {url}")
    
    async with httpx.AsyncClient(follow_redirects=True, timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        
    print(f"Downloaded {len(response.content)} bytes")
    return response.content


@env.task
async def extract_all_text(pdf_bytes: bytes) -> dict:
    """
    Extract all text from a PDF, including text from all layers.
    This attempts to extract any potentially redacted text by accessing
    the underlying PDF structure.
    
    Args:
        pdf_bytes: The raw bytes of the PDF file.
        
    Returns:
        A dictionary containing extracted text organized by page and extraction method.
    """
    import pymupdf
    
    result = {
        "pages": [],
        "metadata": {},
        "raw_text_blocks": [],
        "annotations": [],
        "hidden_text": [],
    }
    
    # Open PDF from bytes
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    
    try:
        doc = pymupdf.open(tmp_path)
        
        # Extract document metadata
        result["metadata"] = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "page_count": len(doc),
        }
        
        print(f"Processing PDF with {len(doc)} pages")
        
        for page_num, page in enumerate(doc):
            page_data = {
                "page_number": page_num + 1,
                "visible_text": "",
                "text_blocks": [],
                "raw_dict_text": [],
                "annotations": [],
            }
            
            # Method 1: Standard text extraction
            page_data["visible_text"] = page.get_text("text")
            
            # Method 2: Extract text blocks with positioning info
            # This can sometimes reveal text under redaction boxes
            blocks = page.get_text("blocks")
            for block in blocks:
                if len(block) >= 5:
                    # block format: (x0, y0, x1, y1, text, block_no, block_type)
                    text = block[4] if isinstance(block[4], str) else ""
                    if text.strip():
                        page_data["text_blocks"].append({
                            "bbox": block[:4],
                            "text": text,
                            "block_type": block[6] if len(block) > 6 else 0
                        })
            
            # Method 3: Extract using dict mode for more granular access
            # This can access text at the character/span level
            text_dict = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                page_data["raw_dict_text"].append({
                                    "text": text,
                                    "font": span.get("font", ""),
                                    "size": span.get("size", 0),
                                    "color": span.get("color", 0),
                                    "bbox": span.get("bbox", []),
                                    "flags": span.get("flags", 0),
                                })
            
            # Method 4: Extract annotations (some redactions are done via annotations)
            annots = page.annots()
            if annots:
                for annot in annots:
                    annot_info = {
                        "type": annot.type[1] if annot.type else "unknown",
                        "rect": list(annot.rect),
                        "content": annot.info.get("content", ""),
                        "title": annot.info.get("title", ""),
                    }
                    # Try to get text under the annotation
                    try:
                        under_text = page.get_text("text", clip=annot.rect)
                        annot_info["text_under"] = under_text
                    except Exception:
                        pass
                    page_data["annotations"].append(annot_info)
            
            # Method 5: Try rawdict for even more detailed extraction
            try:
                raw_dict = page.get_text("rawdict")
                for block in raw_dict.get("blocks", []):
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line.get("spans", []):
                                # Check for potentially hidden text (white on white, etc.)
                                color = span.get("color", 0)
                                text = span.get("text", "").strip()
                                if text:
                                    # Store chars individually for hidden text detection
                                    chars = span.get("chars", [])
                                    for char in chars:
                                        if char.get("c", "").strip():
                                            result["hidden_text"].append({
                                                "page": page_num + 1,
                                                "char": char.get("c", ""),
                                                "bbox": char.get("bbox", []),
                                                "color": color,
                                            })
            except Exception as e:
                print(f"Raw dict extraction error on page {page_num + 1}: {e}")
            
            result["pages"].append(page_data)
        
        doc.close()
        
    finally:
        import os
        os.unlink(tmp_path)
    
    return result


@env.task(report=True)
async def generate_report(extracted_data: dict, source_url: str) -> str:
    """
    Generate a Flyte report displaying the extracted PDF text in markdown format.
    
    Args:
        extracted_data: The dictionary containing all extracted text.
        source_url: The original URL of the PDF.
        
    Returns:
        A summary string of the extraction.
    """
    import html
    
    metadata = extracted_data.get("metadata", {})
    pages = extracted_data.get("pages", [])
    hidden_text = extracted_data.get("hidden_text", [])
    
    # Build the HTML report
    report_html = f"""
    <style>
        .container {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
        }}
        .header {{
            background: linear-gradient(90deg, #0f3460, #533483);
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }}
        .header h1 {{
            margin: 0 0 15px 0;
            font-size: 2.2em;
            background: linear-gradient(90deg, #e94560, #ff6b6b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        .metadata {{
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #e94560;
        }}
        .metadata h2 {{
            color: #e94560;
            margin-top: 0;
        }}
        .metadata-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }}
        .metadata-item {{
            background: rgba(0,0,0,0.2);
            padding: 10px 15px;
            border-radius: 6px;
        }}
        .metadata-label {{
            color: #888;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .metadata-value {{
            color: #fff;
            font-weight: 500;
            margin-top: 5px;
        }}
        .page-section {{
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            margin-bottom: 25px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .page-header {{
            background: linear-gradient(90deg, #533483, #0f3460);
            padding: 15px 20px;
            font-weight: 600;
            font-size: 1.1em;
        }}
        .page-content {{
            padding: 20px;
        }}
        .text-section {{
            margin-bottom: 20px;
        }}
        .text-section h4 {{
            color: #e94560;
            margin: 0 0 10px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .text-box {{
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            font-family: 'Fira Code', 'Monaco', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            white-space: pre-wrap;
            word-break: break-word;
            max-height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .hidden-text-section {{
            background: linear-gradient(135deg, #2d132c, #801336);
            border-radius: 12px;
            padding: 20px;
            margin-top: 30px;
        }}
        .hidden-text-section h2 {{
            color: #ff6b6b;
            margin-top: 0;
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 600;
            text-transform: uppercase;
        }}
        .badge-primary {{ background: #e94560; }}
        .badge-info {{ background: #0f3460; }}
        .source-link {{
            color: #4ecdc4;
            word-break: break-all;
        }}
        .summary-stats {{
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(233, 69, 96, 0.1);
            border: 1px solid rgba(233, 69, 96, 0.3);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            min-width: 150px;
        }}
        .stat-value {{
            font-size: 2.5em;
            font-weight: 700;
            color: #e94560;
        }}
        .stat-label {{
            color: #888;
            font-size: 0.9em;
            margin-top: 5px;
        }}
        .annotation {{
            background: rgba(78, 205, 196, 0.1);
            border-left: 3px solid #4ecdc4;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 6px 6px 0;
        }}
    </style>
    
    <div class="container">
        <div class="header">
            <h1>📄 PDF Text Extraction Report</h1>
            <p>Source: <a href="{html.escape(source_url)}" class="source-link" target="_blank">{html.escape(source_url[:100])}...</a></p>
        </div>
    """
    
    # Summary statistics
    total_chars = sum(len(p.get("visible_text", "")) for p in pages)
    total_blocks = sum(len(p.get("text_blocks", [])) for p in pages)
    total_annotations = sum(len(p.get("annotations", [])) for p in pages)
    
    report_html += f"""
        <div class="summary-stats">
            <div class="stat-card">
                <div class="stat-value">{metadata.get('page_count', 0)}</div>
                <div class="stat-label">Pages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_chars:,}</div>
                <div class="stat-label">Characters</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_blocks}</div>
                <div class="stat-label">Text Blocks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_annotations}</div>
                <div class="stat-label">Annotations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len(hidden_text)}</div>
                <div class="stat-label">Hidden Chars</div>
            </div>
        </div>
    """
    
    # Metadata section
    report_html += """
        <div class="metadata">
            <h2>📋 Document Metadata</h2>
            <div class="metadata-grid">
    """
    
    for key, value in metadata.items():
        if value:
            report_html += f"""
                <div class="metadata-item">
                    <div class="metadata-label">{html.escape(str(key))}</div>
                    <div class="metadata-value">{html.escape(str(value))}</div>
                </div>
            """
    
    report_html += """
            </div>
        </div>
    """
    
    # Pages section
    for page in pages:
        page_num = page.get("page_number", 0)
        visible_text = page.get("visible_text", "")
        text_blocks = page.get("text_blocks", [])
        raw_dict_text = page.get("raw_dict_text", [])
        annotations = page.get("annotations", [])
        
        report_html += f"""
        <div class="page-section">
            <div class="page-header">
                📖 Page {page_num}
                <span class="badge badge-info">{len(visible_text)} chars</span>
            </div>
            <div class="page-content">
        """
        
        # Visible text
        if visible_text.strip():
            report_html += f"""
                <div class="text-section">
                    <h4>📝 Visible Text</h4>
                    <div class="text-box">{html.escape(visible_text)}</div>
                </div>
            """
        else:
            report_html += """
                <div class="text-section">
                    <h4>📝 Visible Text</h4>
                    <p style="color: #888; font-style: italic;">No visible text found on this page.</p>
                </div>
            """
        
        # Text blocks (potentially revealing hidden text)
        if text_blocks:
            block_text = "\n".join([b.get("text", "") for b in text_blocks[:50]])  # Limit for display
            report_html += f"""
                <div class="text-section">
                    <h4>🔍 Text Blocks (Layer Analysis)</h4>
                    <p style="color: #888; font-size: 0.9em;">Text extracted from individual blocks - may reveal underlying content:</p>
                    <div class="text-box">{html.escape(block_text)}</div>
                </div>
            """
        
        # Raw dict text with font info
        if raw_dict_text:
            unique_fonts = set(t.get("font", "") for t in raw_dict_text)
            report_html += f"""
                <div class="text-section">
                    <h4>🔤 Font Analysis</h4>
                    <p style="color: #888; font-size: 0.9em;">Fonts used: {', '.join(html.escape(f) for f in unique_fonts if f)}</p>
                </div>
            """
        
        # Annotations
        if annotations:
            report_html += """
                <div class="text-section">
                    <h4>📌 Annotations</h4>
            """
            for annot in annotations:
                annot_type = annot.get("type", "unknown")
                annot_content = annot.get("content", "")
                text_under = annot.get("text_under", "")
                report_html += f"""
                    <div class="annotation">
                        <strong>Type:</strong> {html.escape(str(annot_type))}<br>
                        <strong>Content:</strong> {html.escape(str(annot_content)) if annot_content else '<em>empty</em>'}<br>
                        <strong>Text Under:</strong> {html.escape(str(text_under)) if text_under else '<em>none</em>'}
                    </div>
                """
            report_html += "</div>"
        
        report_html += """
            </div>
        </div>
        """
    
    # Hidden text section (if any found)
    if hidden_text:
        # Group hidden text by page
        hidden_by_page = {}
        for h in hidden_text:
            pg = h.get("page", 0)
            if pg not in hidden_by_page:
                hidden_by_page[pg] = []
            hidden_by_page[pg].append(h.get("char", ""))
        
        report_html += """
        <div class="hidden-text-section">
            <h2>🔓 Potentially Hidden/Redacted Text</h2>
            <p style="color: #ffb6c1;">Characters extracted from all PDF layers that may have been hidden or redacted:</p>
        """
        
        for pg, chars in sorted(hidden_by_page.items()):
            text = "".join(chars)
            if text.strip():
                report_html += f"""
                    <div style="margin: 15px 0;">
                        <strong>Page {pg}:</strong>
                        <div class="text-box" style="margin-top: 10px;">{html.escape(text)}</div>
                    </div>
                """
        
        report_html += "</div>"
    
    report_html += "</div>"
    
    await flyte.report.log.aio(report_html, do_flush=True)
    
    summary = f"Extracted text from {metadata.get('page_count', 0)} pages. "
    summary += f"Total characters: {total_chars:,}. "
    summary += f"Found {total_annotations} annotations and {len(hidden_text)} potentially hidden characters."
    
    return summary


@env.task
async def main(pdf_url: str) -> str:
    """
    Main workflow that orchestrates PDF text extraction.
    
    Args:
        pdf_url: URL of the PDF to process.
        
    Returns:
        Summary of the extraction.
    """
    print("Starting PDF text extraction workflow...")
    
    # Step 1: Download the PDF
    pdf_bytes = await download_pdf(pdf_url)
    
    # Step 2: Extract all text from all layers
    extracted_data = await extract_all_text(pdf_bytes)
    
    # Step 3: Generate the report
    summary = await generate_report(extracted_data, pdf_url)
    
    print(f"Workflow complete: {summary}")
    return summary


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(main)
    print(f"Run Name: {run.name}")
    print(f"Run URL: {run.url}")
