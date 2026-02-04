import argparse
import html
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate data dictionary files from input_parameters/variables.json."
    )
    parser.add_argument(
        "--variables-config",
        type=Path,
        default=Path("input_parameters") / "variables.json",
        help="Path to variables.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analytics") / "data_analysis" / "artifacts" / "00_documentation",
        help="Directory for generated dictionary files.",
    )
    return parser.parse_args()


def _build_markdown(variables_cfg: dict) -> str:
    dataset_spec = variables_cfg.get("dataset_spec", {})
    variables = dataset_spec.get("variables", [])
    horizon = dataset_spec.get("horizon_months", "n/a")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append("# Data Dictionary")
    lines.append("")
    lines.append(f"- Generated: {timestamp}")
    lines.append(f"- Horizon (months): {horizon}")
    lines.append(f"- Variables: {len(variables)}")
    lines.append("")

    for idx, var in enumerate(variables, start=1):
        name = var.get("name", f"variable_{idx}")
        var_type = var.get("type", "unknown")
        expected_range = var.get("expected_range")
        bands = var.get("bands", [])

        lines.append(f"## {idx}. {name}")
        lines.append("")
        lines.append(f"- Type: `{var_type}`")
        if expected_range:
            min_val = expected_range.get("min", "n/a")
            max_val = expected_range.get("max", "n/a")
            lines.append(f"- Expected range: `{min_val}` to `{max_val}`")
        lines.append("")

        lines.append("| Band | Distribution (%) | Bad Rate Ratio |")
        lines.append("|---|---:|---:|")
        for band in bands:
            band_name = band.get("band", "")
            distribution_pct = band.get("distribution_pct", "")
            bad_rate_ratio = band.get("bad_rate_ratio", "")
            lines.append(
                f"| `{band_name}` | {distribution_pct} | {bad_rate_ratio} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def _build_html(variables_cfg: dict) -> str:
    dataset_spec = variables_cfg.get("dataset_spec", {})
    variables = dataset_spec.get("variables", [])
    horizon = dataset_spec.get("horizon_months", "n/a")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    parts = []
    parts.append("<!doctype html>")
    parts.append("<html><head><meta charset='utf-8'>")
    parts.append("<title>Data Dictionary</title>")
    parts.append(
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;margin:28px;color:#1f2937;}"
        "h1{margin-bottom:6px;} h2{margin-top:28px;border-bottom:1px solid #e5e7eb;padding-bottom:4px;}"
        ".meta{color:#4b5563;margin-bottom:18px;}"
        "table{border-collapse:collapse;width:100%;margin-top:10px;}"
        "th,td{border:1px solid #d1d5db;padding:8px 10px;text-align:left;}"
        "th{background:#f3f4f6;} code{background:#f3f4f6;padding:1px 4px;border-radius:4px;}"
        ".var-meta{margin:6px 0 10px 0;}"
        "</style>"
    )
    parts.append("</head><body>")
    parts.append("<h1>Data Dictionary</h1>")
    parts.append(
        f"<div class='meta'>Generated: {html.escape(timestamp)} | Horizon (months): {html.escape(str(horizon))} | Variables: {len(variables)}</div>"
    )

    for idx, var in enumerate(variables, start=1):
        name = html.escape(str(var.get("name", f"variable_{idx}")))
        var_type = html.escape(str(var.get("type", "unknown")))
        expected_range = var.get("expected_range")
        bands = var.get("bands", [])

        parts.append(f"<h2>{idx}. {name}</h2>")
        parts.append("<div class='var-meta'>")
        parts.append(f"Type: <code>{var_type}</code><br>")
        if expected_range:
            min_val = html.escape(str(expected_range.get("min", "n/a")))
            max_val = html.escape(str(expected_range.get("max", "n/a")))
            parts.append(f"Expected range: <code>{min_val}</code> to <code>{max_val}</code>")
        parts.append("</div>")

        parts.append("<table>")
        parts.append("<thead><tr><th>Band</th><th>Distribution (%)</th><th>Bad Rate Ratio</th></tr></thead>")
        parts.append("<tbody>")
        for band in bands:
            band_name = html.escape(str(band.get("band", "")))
            distribution_pct = html.escape(str(band.get("distribution_pct", "")))
            bad_rate_ratio = html.escape(str(band.get("bad_rate_ratio", "")))
            parts.append(
                f"<tr><td><code>{band_name}</code></td><td>{distribution_pct}</td><td>{bad_rate_ratio}</td></tr>"
            )
        parts.append("</tbody></table>")

    parts.append("</body></html>")
    return "".join(parts)


def _try_write_pdf_reportlab(variables_cfg: dict, pdf_path: Path) -> bool:
    try:
        from reportlab.lib.pagesizes import A4  # type: ignore
        from reportlab.pdfgen import canvas  # type: ignore
    except Exception:
        return False

    dataset_spec = variables_cfg.get("dataset_spec", {})
    variables = dataset_spec.get("variables", [])
    horizon = dataset_spec.get("horizon_months", "n/a")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4
    margin = 36
    y = height - margin

    def new_page() -> None:
        nonlocal y
        c.showPage()
        y = height - margin

    def write_line(text: str, size: int = 10, gap: int = 14) -> None:
        nonlocal y
        if y < margin + 20:
            new_page()
        c.setFont("Helvetica", size)
        c.drawString(margin, y, text[:160])
        y -= gap

    write_line("Data Dictionary", size=16, gap=18)
    write_line(f"Generated: {timestamp}")
    write_line(f"Horizon (months): {horizon}")
    write_line(f"Variables: {len(variables)}")
    y -= 6

    for idx, var in enumerate(variables, start=1):
        name = str(var.get("name", f"variable_{idx}"))
        var_type = str(var.get("type", "unknown"))
        expected_range = var.get("expected_range")
        bands = var.get("bands", [])

        write_line(f"{idx}. {name}", size=12, gap=16)
        write_line(f"Type: {var_type}")
        if expected_range:
            write_line(
                f"Expected range: {expected_range.get('min', 'n/a')} to {expected_range.get('max', 'n/a')}"
            )
        write_line("Band | Distribution (%) | Bad Rate Ratio")
        write_line("-" * 70)
        for band in bands:
            band_name = str(band.get("band", ""))
            distribution_pct = str(band.get("distribution_pct", ""))
            bad_rate_ratio = str(band.get("bad_rate_ratio", ""))
            write_line(f"{band_name} | {distribution_pct} | {bad_rate_ratio}")
        y -= 6

    c.save()
    return True


def _try_write_pdf_weasy(html_path: Path, pdf_path: Path) -> bool:
    try:
        from weasyprint import HTML  # type: ignore
    except Exception:
        return False

    HTML(filename=str(html_path)).write_pdf(str(pdf_path))
    return True


def main() -> None:
    args = parse_args()
    if not args.variables_config.exists():
        raise FileNotFoundError(f"Missing variables config: {args.variables_config}")

    variables_cfg = json.loads(args.variables_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    md_path = args.output_dir / "data_dictionary.md"
    html_path = args.output_dir / "data_dictionary.html"
    pdf_path = args.output_dir / "data_dictionary.pdf"

    markdown = _build_markdown(variables_cfg)
    html_doc = _build_html(variables_cfg)

    md_path.write_text(markdown, encoding="utf-8")
    html_path.write_text(html_doc, encoding="utf-8")
    pdf_ok = _try_write_pdf_reportlab(variables_cfg, pdf_path)
    pdf_method = "reportlab"
    if not pdf_ok:
        pdf_ok = _try_write_pdf_weasy(html_path, pdf_path)
        pdf_method = "weasyprint"

    print(f"Generated: {md_path}")
    print(f"Generated: {html_path}")
    if pdf_ok:
        print(f"Generated: {pdf_path} (via {pdf_method})")
    else:
        print("PDF generation skipped (install reportlab or weasyprint).")


if __name__ == "__main__":
    main()
