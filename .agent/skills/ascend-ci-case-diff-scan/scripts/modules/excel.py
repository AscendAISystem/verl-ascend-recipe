#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Excel report rendering for the Ascend CI case diff scanner."""

from __future__ import annotations

from itertools import zip_longest
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr
from zipfile import ZIP_DEFLATED, ZipFile

DEFAULT_COLUMN_WIDTH = 18
IGNORED_WORKFLOW_WIDTHS = (70,)
SCANNED_WORKFLOW_WIDTHS = (90, 22, 26)
CASE_DETAIL_WIDTHS = (64, 16, 52, 14, 56, 72, 88, 52, 14, 56, 72, 88)
PAST_SUMMARY_WIDTHS = (42, 16, 12, 12, 12)
PAST_WORKFLOW_WIDTHS = (48, 16, 12, 12, 12, 12, 32)
PAST_CASE_WIDTHS = (48, 16, 46, 26, 20, 54, 72, 18, 64, 18, 54)
PAST_DETAIL_WIDTHS = (42, 24, 36, 42)
CASE_STATUS_LABELS = (
    ("matched", "Matched"),
    ("cpu_gpu_only", "CPU/GPU Only"),
    ("npu_only", "NPU Only"),
    ("manual_review", "Manual Review"),
)


def write_excel_report(path: Path, report: dict) -> None:
    """Write the scan report as a minimal XLSX workbook."""
    sheets = [
        ("Ignored Workflows", _ignored_workflow_rows(report), IGNORED_WORKFLOW_WIDTHS),
        ("Scanned Workflows", _scanned_workflow_rows(report), SCANNED_WORKFLOW_WIDTHS),
        ("UT Cases", _case_rows(report["ut_details"], "UT Case Name"), CASE_DETAIL_WIDTHS),
        ("ST Cases", _case_rows(report["st_details"], "ST Case Name"), CASE_DETAIL_WIDTHS),
    ]
    _write_workbook(path, sheets)


def write_past_commit_excel_report(path: Path, report: dict) -> None:
    """Write the past-N-days report as a minimal XLSX workbook."""
    sheets = [
        ("Summary", _past_summary_rows(report), PAST_SUMMARY_WIDTHS),
        ("Changed Workflows", _past_workflow_rows(report), PAST_WORKFLOW_WIDTHS),
        ("Changed Cases", _past_case_rows(report), PAST_CASE_WIDTHS),
        ("Commit Details", _past_detail_rows(report), PAST_DETAIL_WIDTHS),
    ]
    _write_workbook(path, sheets)


def _ignored_workflow_rows(report: dict) -> list[list[str]]:
    rows = [["Workflow Name"]]
    rows.extend([[path] for path in report["ignored_workflows"]])
    return rows


def _scanned_workflow_rows(report: dict) -> list[list[object]]:
    rows: list[list[object]] = [["Workflow Name", "CPU/GPU Case Count", "NPU Supported Case Count"]]
    rows.extend(
        [
            _excel_multiline(row["workflow_name"]),
            row["cpu_gpu_case_count"],
            row["npu_supported_case_count"],
        ]
        for row in report["scanned_workflows"]
    )
    return rows


def _case_rows(details: dict, case_header: str) -> list[list[object]]:
    rows = [
        [
            case_header,
            "Match Status",
            "CPU/GPU Workflow Name",
            "CPU/GPU Line Number",
            "CPU/GPU Workflow Context Name",
            "CPU/GPU Signature",
            "CPU/GPU Full Raw Command",
            "NPU Workflow Name",
            "NPU Line Number",
            "NPU Workflow Context Name",
            "NPU Signature",
            "NPU Full Raw Command",
        ]
    ]
    for section_key, status in CASE_STATUS_LABELS:
        for item in details[section_key]:
            rows.extend(_case_item_rows(item, status))
    return rows


def _case_item_rows(item: dict, status: str) -> list[list[object]]:
    rows = []
    cpu_gpu_refs = sorted(item["cpu_gpu_refs"], key=_ref_sort_key)
    npu_refs = sorted(item["npu_refs"], key=_ref_sort_key)
    for cpu_gpu_ref, npu_ref in zip_longest(cpu_gpu_refs, npu_refs):
        rows.append([item["name"], status, *_side_cells(cpu_gpu_ref), *_side_cells(npu_ref)])
    return rows


def _ref_sort_key(ref: dict) -> tuple[object, ...]:
    return ref["name"], ref["workflow_path"], ref["line_number"]


def _side_cells(ref: dict | None) -> list[object]:
    if not ref:
        return ["", "", "", ""]
    return [
        ref["workflow_path"],
        ref["line_number"],
        f"{ref['workflow_name']} / {ref['job_name']} / {ref['step_name']}",
        ref["signature"],
        ref["raw_command"],
    ]


def _write_workbook(path: Path, sheets: list[tuple[str, list[list[object]], tuple[int, ...]]]) -> None:
    with ZipFile(path, "w", ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", _content_types_xml(len(sheets)))
        workbook.writestr("_rels/.rels", _root_rels_xml())
        workbook.writestr("xl/workbook.xml", _workbook_xml(sheets))
        workbook.writestr("xl/_rels/workbook.xml.rels", _workbook_rels_xml(len(sheets)))
        workbook.writestr("xl/styles.xml", _styles_xml())
        for index, (_, rows, column_widths) in enumerate(sheets, start=1):
            workbook.writestr(f"xl/worksheets/sheet{index}.xml", _worksheet_xml(rows, column_widths))


def _content_types_xml(sheet_count: int) -> str:
    sheet_overrides = "\n".join(
        (
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
        for index in range(1, sheet_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
{_override_xml("/xl/workbook.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml")}
{_override_xml("/xl/styles.xml", "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml")}
{sheet_overrides}
</Types>"""


def _root_rels_xml() -> str:
    return (
        """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
"""
        + _relationship_xml(
            "rId1",
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
            "xl/workbook.xml",
        )
        + """
</Relationships>"""
    )


def _workbook_xml(sheets: list[tuple[str, list[list[object]], tuple[int, ...]]]) -> str:
    sheet_xml = "\n".join(
        f'<sheet name={quoteattr(name)} sheetId="{index}" r:id="rId{index}"/>'
        for index, (name, _, _) in enumerate(sheets, start=1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook {_workbook_namespace_attrs()}>
<sheets>
{sheet_xml}
</sheets>
</workbook>"""


def _workbook_rels_xml(sheet_count: int) -> str:
    sheet_rels = "\n".join(
        (
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
        for index in range(1, sheet_count + 1)
    )
    style_id = sheet_count + 1
    style_rel = _relationship_xml(
        f"rId{style_id}",
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles",
        "styles.xml",
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
{sheet_rels}
{style_rel}
</Relationships>"""


def _override_xml(part_name: str, content_type: str) -> str:
    return f'<Override PartName="{part_name}" ContentType="{content_type}"/>'


def _relationship_xml(relation_id: str, relation_type: str, target: str) -> str:
    return f'<Relationship Id="{relation_id}" Type="{relation_type}" Target="{target}"/>'


def _workbook_namespace_attrs() -> str:
    main_xmlns = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'
    rels_xmlns = 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"'
    return f"{main_xmlns} {rels_xmlns}"


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="2">
<font><sz val="11"/><name val="Calibri"/></font>
<font><b/><sz val="11"/><name val="Calibri"/></font>
</fonts>
<fills count="3">
<fill><patternFill patternType="none"/></fill>
<fill><patternFill patternType="gray125"/></fill>
<fill><patternFill patternType="solid"><fgColor rgb="FFD9D9D9"/><bgColor indexed="64"/></patternFill></fill>
</fills>
<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="3">
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
<xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0" applyAlignment="1">
<alignment wrapText="1" vertical="top"/></xf>
<xf numFmtId="0" fontId="1" fillId="2" borderId="0" xfId="0" applyFont="1" applyFill="1"
applyAlignment="1"><alignment wrapText="1" vertical="top"/></xf>
</cellXfs>
</styleSheet>"""


def _worksheet_xml(rows: list[list[object]], column_widths: tuple[int, ...]) -> str:
    max_columns = max((len(row) for row in rows), default=1)
    column_xml = "".join(
        f'<col min="{index}" max="{index}" width="{_column_width(index, column_widths)}" customWidth="1"/>'
        for index in range(1, max_columns + 1)
    )
    row_xml = "\n".join(_row_xml(row_index, row) for row_index, row in enumerate(rows, start=1))
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<cols>{column_xml}</cols>
<sheetData>
{row_xml}
</sheetData>
</worksheet>"""


def _row_xml(row_index: int, row: list[object]) -> str:
    cells = "".join(_cell_xml(row_index, column_index, value) for column_index, value in enumerate(row, start=1))
    return f'<row r="{row_index}">{cells}</row>'


def _cell_xml(row_index: int, column_index: int, value: object) -> str:
    cell_ref = f"{_column_name(column_index)}{row_index}"
    style_id = 2 if row_index == 1 else 1
    if isinstance(value, int):
        return f'<c r="{cell_ref}" s="{style_id}"><v>{value}</v></c>'
    text = _xml_text(str(value)) if value is not None else ""
    return f'<c r="{cell_ref}" s="{style_id}" t="inlineStr"><is><t>{text}</t></is></c>'


def _column_name(index: int) -> str:
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _column_width(index: int, column_widths: tuple[int, ...]) -> int:
    if index <= len(column_widths):
        return column_widths[index - 1]
    return DEFAULT_COLUMN_WIDTH


def _excel_multiline(value: str) -> str:
    return value.replace("<br>", "\n")


def _xml_text(value: str) -> str:
    cleaned = "".join(character for character in value if _is_valid_xml_char(character))
    return escape(cleaned)


def _past_summary_rows(report: dict) -> list[list[object]]:
    rows = [
        [
            "Workflow",
            "UT Not Fully Aligned with NPU Count",
            "ST Not Fully Aligned with NPU Count",
            "Change Count",
            "Related Commits",
        ]
    ]
    rows.extend(
        [
            row["workflow_path"],
            row["ut_gap_count"],
            row["st_gap_count"],
            len(row["commit_hashes"]),
            ", ".join(commit[:12] for commit in row["commit_hashes"]),
        ]
        for row in report["workflow_changes"]
    )
    return rows


def _past_workflow_rows(report: dict) -> list[list[object]]:
    rows = [
        [
            "Workflow",
            "Change",
            "Window Start Cases",
            "Current HEAD Cases",
            "UT Not Fully Aligned with NPU",
            "ST Not Fully Aligned with NPU",
            "Related Commits",
        ]
    ]
    rows.extend(
        [
            row["workflow_path"],
            row["workflow_status"],
            row["case_count_base"],
            row["case_count_head"],
            row["ut_gap_count"],
            row["st_gap_count"],
            ", ".join(commit[:12] for commit in row["commit_hashes"]),
        ]
        for row in report["workflow_changes"]
    )
    return rows


def _past_case_rows(report: dict) -> list[list[object]]:
    rows = [
        [
            "Workflow",
            "Case Name",
            "Kind",
            "Target",
            "Line",
            "Workflow Context",
            "Signature",
            "NPU Status",
            "Related Commits",
            "NPU Refs",
        ]
    ]
    rows.extend(
        [
            row["workflow_path"],
            row["case_name"],
            row["case_kind"],
            row["target"],
            row["line_number"],
            row["workflow_context"],
            row["signature"],
            row["npu_status"],
            ", ".join(commit[:12] for commit in row["commit_hashes"]),
            _excel_multiline(
                "\n".join(
                    f"{ref['workflow_name']} / {ref['job_name']} / "
                    f"{ref['step_name']} {ref['workflow_path']}:{ref['line_number']}"
                    for ref in row["npu_refs"]
                )
            )
            if row["npu_refs"]
            else "",
        ]
        for row in report["case_details"]
    )
    return rows


def _past_detail_rows(report: dict) -> list[list[object]]:
    rows = [
        [
            "Commit Hash",
            "Commit Time",
            "Commit Title",
            "Affected Workflows",
        ]
    ]
    for row in report["commit_details"]:
        rows.append(
            [
                row["commit_hash"],
                row["commit_time"],
                row["commit_title"],
                _excel_multiline("\n".join(row["affected_workflows"])),
            ]
        )
    return rows


def _is_valid_xml_char(character: str) -> bool:
    codepoint = ord(character)
    return codepoint in (0x9, 0xA, 0xD) or 0x20 <= codepoint <= 0xD7FF or 0xE000 <= codepoint <= 0xFFFD
