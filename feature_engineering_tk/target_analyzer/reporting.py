"""
Reporting mixin for TargetAnalyzer.

Contains report generation and export methods.
"""

import pandas as pd
import json
import logging
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)


class ReportingMixin:
    """
    Mixin providing report generation and export for TargetAnalyzer.

    Requires: self.df, self.target_column, self.task,
              self.get_task_info, self.analyze_class_distribution, self.get_class_imbalance_info,
              self.analyze_class_wise_statistics, self.analyze_target_distribution,
              self.analyze_feature_correlations, self.analyze_feature_target_relationship,
              self.analyze_mutual_information, self.analyze_data_quality, self.calculate_vif,
              self.generate_recommendations
    """

    def generate_full_report(self) -> Dict[str, Any]:
        """
        Generate complete analysis report with all metrics in structured format.

        Returns:
            Dict containing all analysis results:
            - task_info: Basic task information
            - distribution: Class or target distribution
            - imbalance: Class imbalance info (classification only)
            - relationships: Feature-target relationships
            - class_stats: Class-wise statistics (classification only)
            - correlations: Feature correlations (regression only)
            - mutual_info: Mutual information scores
            - data_quality: Data quality metrics
            - vif: Variance Inflation Factors
            - recommendations: List of actionable recommendations
        """
        report = {
            'task': self.task,
            'task_info': self.get_task_info(),
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Task-specific distributions
        if self.task == 'classification':
            report['distribution'] = self.analyze_class_distribution().to_dict('records')
            report['imbalance'] = self.get_class_imbalance_info()
            report['class_stats'] = {
                feature: df.to_dict('records')
                for feature, df in self.analyze_class_wise_statistics().items()
            }
        else:
            report['distribution'] = self.analyze_target_distribution()
            report['imbalance'] = None
            report['class_stats'] = None

            # Regression-specific
            corr_df = self.analyze_feature_correlations()
            report['correlations'] = corr_df.to_dict('records') if not corr_df.empty else []

        # Common analyses
        rel_df = self.analyze_feature_target_relationship()
        report['relationships'] = rel_df.to_dict('records') if not rel_df.empty else []

        mi_df = self.analyze_mutual_information()
        report['mutual_info'] = mi_df.to_dict('records') if not mi_df.empty else []

        report['data_quality'] = self.analyze_data_quality()

        vif_df = self.calculate_vif()
        report['vif'] = vif_df.to_dict('records') if not vif_df.empty else []

        report['recommendations'] = self.generate_recommendations()

        return report

    def export_report(self, filepath: str, format: str = 'html') -> None:
        """
        Export comprehensive analysis report to file.

        Args:
            filepath: Path to save the report
            format: Export format ('html', 'markdown', 'json')

        Raises:
            ValueError: If format is not supported
        """
        if format not in ['html', 'markdown', 'json']:
            raise ValueError(f"Format must be 'html', 'markdown', or 'json', got '{format}'")

        report_data = self.generate_full_report()

        if format == 'json':
            self._export_json(filepath, report_data)
        elif format == 'markdown':
            self._export_markdown(filepath, report_data)
        elif format == 'html':
            self._export_html(filepath, report_data)

        logger.info(f"Report exported to: {filepath}")

    def _export_json(self, filepath: str, report_data: Dict[str, Any]) -> None:
        """Export report as JSON."""
        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

    def _export_markdown(self, filepath: str, report_data: Dict[str, Any]) -> None:
        """Export report as Markdown with table of contents and callout boxes."""
        lines = []

        # Header
        lines.append("# Target Analysis Report")
        lines.append("")
        lines.append(f"**Generated**: {report_data['timestamp']}")
        lines.append(f"**Task Type**: {report_data['task'].upper()}")
        lines.append(f"**Target Column**: {report_data['task_info']['target_column']}")
        lines.append("")

        # Table of Contents
        lines.append("## Table of Contents")
        lines.append("")
        lines.append("- [Task Information](#task-information)")
        lines.append("- [Distribution Analysis](#distribution-analysis)")
        if report_data['relationships']:
            lines.append("- [Feature-Target Relationships](#feature-target-relationships)")
        if report_data['data_quality']:
            lines.append("- [Data Quality](#data-quality)")
        if report_data['vif']:
            lines.append("- [Multicollinearity (VIF)](#multicollinearity-vif)")
        lines.append("- [Recommendations](#recommendations)")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Task Info
        lines.append("## Task Information")
        lines.append("")
        lines.append(f"| Property | Value |")
        lines.append("|----------|-------|")
        lines.append(f"| Data Type | {report_data['task_info']['target_dtype']} |")
        lines.append(f"| Unique Values | {report_data['task_info']['unique_values']} |")
        lines.append(f"| Missing Values | {report_data['task_info']['missing_count']} ({report_data['task_info']['missing_percent']:.2f}%) |")

        if report_data['task'] == 'classification':
            lines.append(f"| Number of Classes | {report_data['task_info']['class_count']} |")
            lines.append(f"| Classes | {', '.join(map(str, report_data['task_info']['classes']))} |")

        lines.append("")
        lines.append("---")
        lines.append("")

        # Distribution
        lines.append("## Distribution Analysis")
        lines.append("")
        if report_data['task'] == 'classification' and report_data['distribution']:
            lines.append("### Class Distribution")
            lines.append("")
            lines.append("| Class | Count | Percentage | Imbalance Ratio |")
            lines.append("|-------|------:|----------:|--------------:|")
            for item in report_data['distribution']:
                lines.append(f"| {item['class']} | {item['count']} | {item['percentage']:.2f}% | {item['imbalance_ratio']:.2f} |")
            lines.append("")

            if report_data['imbalance']:
                severity = report_data['imbalance']['severity']
                if severity == 'severe':
                    lines.append("> **Warning**: Severe class imbalance detected!")
                    lines.append(">")
                    lines.append(f"> - **Severity**: {severity.upper()}")
                    lines.append(f"> - **Recommendation**: {report_data['imbalance']['recommendation']}")
                elif severity == 'moderate':
                    lines.append("> **Note**: Moderate class imbalance detected.")
                    lines.append(">")
                    lines.append(f"> - **Severity**: {severity.upper()}")
                    lines.append(f"> - **Recommendation**: {report_data['imbalance']['recommendation']}")
                else:
                    lines.append("> **OK**: Classes are well balanced.")
                lines.append("")

        elif report_data['task'] == 'regression' and report_data['distribution']:
            lines.append("### Target Statistics")
            lines.append("")
            dist = report_data['distribution']
            lines.append("| Statistic | Value |")
            lines.append("|-----------|------:|")
            lines.append(f"| Mean | {dist['mean']:.4f} |")
            lines.append(f"| Median | {dist['median']:.4f} |")
            lines.append(f"| Std Dev | {dist['std']:.4f} |")
            lines.append(f"| Skewness | {dist['skewness']:.4f} |")
            lines.append(f"| Kurtosis | {dist['kurtosis']:.4f} |")
            lines.append("")

            # Add skewness warning
            skew = dist.get('skewness', 0)
            if abs(skew) > 2:
                lines.append("> **Warning**: Target is highly skewed (skewness = {:.2f})".format(skew))
                lines.append(">")
                lines.append("> Consider applying log transformation before modeling.")
            elif abs(skew) > 1:
                lines.append("> **Note**: Target is moderately skewed (skewness = {:.2f})".format(skew))
                lines.append(">")
                lines.append("> Transformation may improve model performance.")
            lines.append("")

        lines.append("---")
        lines.append("")

        # Relationships
        if report_data['relationships']:
            lines.append("## Feature-Target Relationships")
            lines.append("")
            lines.append("### Top 10 Most Significant Features")
            lines.append("")
            lines.append("| Feature | Test Type | Statistic | P-Value | Significant |")
            lines.append("|---------|-----------|----------:|--------:|:-----------:|")
            for item in report_data['relationships'][:10]:
                sig = "Yes" if item['significant'] else "No"
                lines.append(f"| {item['feature']} | {item['test_type']} | {item['statistic']:.4f} | {item['pvalue']:.4e} | {sig} |")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Data Quality
        if report_data['data_quality']:
            lines.append("## Data Quality")
            lines.append("")
            quality = report_data['data_quality']

            if quality['missing_values']:
                lines.append("### Features with Missing Values")
                lines.append("")
                lines.append("| Feature | Missing Count | Missing % |")
                lines.append("|---------|-------------:|----------:|")
                for feat, info in quality['missing_values'].items():
                    lines.append(f"| {feat} | {info['count']} | {info['percent']:.2f}% |")
                lines.append("")
            else:
                lines.append("> **OK**: No missing values detected.")
                lines.append("")

            if quality['constant_features']:
                lines.append("### Constant Features")
                lines.append("")
                lines.append("> **Warning**: The following features have constant values and should be removed:")
                lines.append(">")
                for feat in quality['constant_features']:
                    lines.append(f"> - {feat}")
                lines.append("")

            if quality['leakage_suspects']:
                lines.append("### Potential Data Leakage")
                lines.append("")
                lines.append("> **Warning**: Potential data leakage detected!")
                lines.append(">")
                for suspect in quality['leakage_suspects']:
                    lines.append(f"> - **{suspect['feature']}**: {suspect['reason']} (Severity: {suspect['severity']})")
                lines.append("")

            lines.append("---")
            lines.append("")

        # VIF
        if report_data['vif']:
            lines.append("## Multicollinearity (VIF)")
            lines.append("")
            high_vif = [item for item in report_data['vif'] if item['VIF'] > 10]
            if high_vif:
                lines.append("> **Warning**: High multicollinearity detected in the following features:")
                lines.append("")
                lines.append("| Feature | VIF |")
                lines.append("|---------|----:|")
                for item in high_vif:
                    lines.append(f"| {item['feature']} | {item['VIF']:.2f} |")
                lines.append("")
                lines.append("*Features with VIF > 10 indicate high multicollinearity. Consider removing or combining these features.*")
            else:
                lines.append("> **OK**: No features with high multicollinearity detected (VIF < 10).")
            lines.append("")
            lines.append("---")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")
        if report_data['recommendations']:
            for i, rec in enumerate(report_data['recommendations'], 1):
                lines.append(f"{i}. {rec}")
        else:
            lines.append("No specific recommendations at this time.")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by Feature Engineering Toolkit*")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _export_html(self, filepath: str, report_data: Dict[str, Any]) -> None:
        """Export report as HTML with collapsible sections and improved styling."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Target Analysis Report</title>
    <style>
        :root {{
            --primary: #3498db;
            --secondary: #2c3e50;
            --success: #28a745;
            --warning: #ffc107;
            --danger: #dc3545;
            --light: #f8f9fa;
            --border: #dee2e6;
        }}
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: var(--secondary);
            border-bottom: 3px solid var(--primary);
            padding-bottom: 15px;
            margin-bottom: 20px;
        }}
        /* Collapsible sections */
        details {{
            margin: 15px 0;
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }}
        summary {{
            background: var(--light);
            padding: 15px 20px;
            cursor: pointer;
            font-weight: 600;
            font-size: 1.1em;
            color: var(--secondary);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        summary:hover {{ background: #e9ecef; }}
        summary::after {{
            content: '+';
            font-size: 1.2em;
            font-weight: bold;
            color: var(--primary);
        }}
        details[open] summary::after {{ content: '-'; }}
        details[open] summary {{ border-bottom: 1px solid var(--border); }}
        .section-content {{ padding: 20px; }}
        /* Tables */
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 0.95em;
        }}
        th, td {{
            border: 1px solid var(--border);
            padding: 10px 12px;
            text-align: left;
        }}
        th {{
            background-color: var(--primary);
            color: white;
            font-weight: 600;
            position: sticky;
            top: 0;
        }}
        tr:nth-child(even) {{ background-color: #f8f9fa; }}
        tr:hover {{ background-color: #e9ecef; }}
        td.number {{ text-align: right; font-family: monospace; }}
        /* Alert boxes */
        .info-box {{
            background-color: #e7f3ff;
            border-left: 4px solid var(--primary);
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        .warning-box {{
            background-color: #fff3cd;
            border-left: 4px solid var(--warning);
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        .success-box {{
            background-color: #d4edda;
            border-left: 4px solid var(--success);
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        .danger-box {{
            background-color: #f8d7da;
            border-left: 4px solid var(--danger);
            padding: 15px 20px;
            margin: 15px 0;
            border-radius: 0 8px 8px 0;
        }}
        /* Recommendations */
        .recommendation {{
            padding: 12px 15px;
            margin: 8px 0;
            background-color: var(--light);
            border-radius: 6px;
            border-left: 3px solid var(--primary);
        }}
        .recommendation:hover {{ background-color: #e9ecef; }}
        /* Badge styles */
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            font-weight: 600;
        }}
        .badge-success {{ background: #d4edda; color: #155724; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        /* Quick stats */
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: var(--light);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid var(--border);
        }}
        .stat-value {{ font-size: 1.5em; font-weight: bold; color: var(--primary); }}
        .stat-label {{ font-size: 0.9em; color: #666; margin-top: 5px; }}
        /* Print styles */
        @media print {{
            details {{ border: none; }}
            summary {{ background: none; }}
            details[open] summary {{ border-bottom: 1px solid #ccc; }}
            .section-content {{ padding: 10px 0; }}
        }}
    </style>
</head>
<body>
    <h1>Target Analysis Report</h1>

    <div class="info-box">
        <strong>Generated:</strong> {report_data['timestamp']}<br>
        <strong>Task Type:</strong> {report_data['task'].upper()}<br>
        <strong>Target Column:</strong> {report_data['task_info']['target_column']}
    </div>
"""

        # Quick stats grid
        html += '<div class="stats-grid">'
        html += f'<div class="stat-card"><div class="stat-value">{report_data["task_info"]["unique_values"]}</div><div class="stat-label">Unique Values</div></div>'
        html += f'<div class="stat-card"><div class="stat-value">{report_data["task_info"]["missing_percent"]:.1f}%</div><div class="stat-label">Missing</div></div>'
        if report_data['task'] == 'classification':
            html += f'<div class="stat-card"><div class="stat-value">{report_data["task_info"]["class_count"]}</div><div class="stat-label">Classes</div></div>'
        html += '</div>'

        # Task Information (collapsible)
        html += '<details open>'
        html += '<summary>Task Information</summary>'
        html += '<div class="section-content">'
        html += '<table>'
        html += '<tr><th>Property</th><th>Value</th></tr>'
        html += f'<tr><td>Data Type</td><td>{report_data["task_info"]["target_dtype"]}</td></tr>'
        html += f'<tr><td>Unique Values</td><td class="number">{report_data["task_info"]["unique_values"]}</td></tr>'
        html += f'<tr><td>Missing Values</td><td class="number">{report_data["task_info"]["missing_count"]} ({report_data["task_info"]["missing_percent"]:.2f}%)</td></tr>'
        if report_data['task'] == 'classification':
            html += f'<tr><td>Number of Classes</td><td class="number">{report_data["task_info"]["class_count"]}</td></tr>'
            html += f'<tr><td>Classes</td><td>{", ".join(map(str, report_data["task_info"]["classes"]))}</td></tr>'
        html += '</table>'
        html += '</div></details>'

        # Distribution (collapsible)
        if report_data['task'] == 'classification' and report_data['distribution']:
            html += '<details open>'
            html += '<summary>Class Distribution</summary>'
            html += '<div class="section-content">'
            html += '<table>'
            html += '<tr><th>Class</th><th>Count</th><th>Percentage</th><th>Imbalance Ratio</th></tr>'
            for item in report_data['distribution']:
                html += f'<tr><td>{item["class"]}</td><td class="number">{item["count"]}</td><td class="number">{item["percentage"]:.2f}%</td><td class="number">{item["imbalance_ratio"]:.2f}</td></tr>'
            html += '</table>'

            if report_data['imbalance']:
                severity = report_data['imbalance']['severity']
                if severity == 'severe':
                    html += f'<div class="danger-box"><strong>Severe Imbalance:</strong> {report_data["imbalance"]["recommendation"]}</div>'
                elif severity == 'moderate':
                    html += f'<div class="warning-box"><strong>Moderate Imbalance:</strong> {report_data["imbalance"]["recommendation"]}</div>'
                else:
                    html += '<div class="success-box"><strong>Balanced:</strong> Classes are well balanced.</div>'
            html += '</div></details>'

        elif report_data['task'] == 'regression' and report_data['distribution']:
            dist = report_data['distribution']
            html += '<details open>'
            html += '<summary>Target Distribution</summary>'
            html += '<div class="section-content">'
            html += '<table>'
            html += '<tr><th>Statistic</th><th>Value</th></tr>'
            html += f'<tr><td>Mean</td><td class="number">{dist["mean"]:.4f}</td></tr>'
            html += f'<tr><td>Median</td><td class="number">{dist["median"]:.4f}</td></tr>'
            html += f'<tr><td>Std Dev</td><td class="number">{dist["std"]:.4f}</td></tr>'
            html += f'<tr><td>Min</td><td class="number">{dist["min"]:.4f}</td></tr>'
            html += f'<tr><td>Max</td><td class="number">{dist["max"]:.4f}</td></tr>'
            html += f'<tr><td>Skewness</td><td class="number">{dist["skewness"]:.4f}</td></tr>'
            html += f'<tr><td>Kurtosis</td><td class="number">{dist["kurtosis"]:.4f}</td></tr>'
            html += '</table>'

            skew = dist.get('skewness', 0)
            if abs(skew) > 2:
                html += f'<div class="warning-box"><strong>Highly Skewed:</strong> Consider log transformation (skewness = {skew:.2f})</div>'
            elif abs(skew) > 1:
                html += f'<div class="info-box"><strong>Moderately Skewed:</strong> Transformation may improve results (skewness = {skew:.2f})</div>'
            html += '</div></details>'

        # Relationships (collapsible)
        if report_data['relationships']:
            html += '<details>'
            html += '<summary>Feature-Target Relationships (Top 10)</summary>'
            html += '<div class="section-content">'
            html += '<table>'
            html += '<tr><th>Feature</th><th>Test Type</th><th>Statistic</th><th>P-Value</th><th>Significant</th></tr>'
            for item in report_data['relationships'][:10]:
                sig_badge = '<span class="badge badge-success">Yes</span>' if item['significant'] else '<span class="badge badge-danger">No</span>'
                html += f'<tr><td>{item["feature"]}</td><td>{item["test_type"]}</td><td class="number">{item["statistic"]:.4f}</td><td class="number">{item["pvalue"]:.4e}</td><td>{sig_badge}</td></tr>'
            html += '</table>'
            html += '</div></details>'

        # Data Quality (collapsible)
        if report_data['data_quality']:
            quality = report_data['data_quality']
            html += '<details>'
            html += '<summary>Data Quality</summary>'
            html += '<div class="section-content">'

            if quality['missing_values']:
                html += '<h4>Missing Values</h4>'
                html += '<table>'
                html += '<tr><th>Feature</th><th>Count</th><th>Percent</th></tr>'
                for feat, info in quality['missing_values'].items():
                    html += f'<tr><td>{feat}</td><td class="number">{info["count"]}</td><td class="number">{info["percent"]:.2f}%</td></tr>'
                html += '</table>'
            else:
                html += '<div class="success-box">No missing values detected.</div>'

            if quality['constant_features']:
                html += '<div class="warning-box"><strong>Constant Features (remove these):</strong> ' + ', '.join(quality['constant_features']) + '</div>'

            if quality['leakage_suspects']:
                html += '<div class="danger-box"><strong>Potential Data Leakage:</strong><ul>'
                for suspect in quality['leakage_suspects']:
                    html += f'<li><strong>{suspect["feature"]}:</strong> {suspect["reason"]} (Severity: {suspect["severity"]})</li>'
                html += '</ul></div>'

            html += '</div></details>'

        # VIF (collapsible)
        if report_data['vif']:
            html += '<details>'
            html += '<summary>Multicollinearity (VIF)</summary>'
            html += '<div class="section-content">'
            high_vif = [item for item in report_data['vif'] if item['VIF'] > 10]
            if high_vif:
                html += '<div class="warning-box"><strong>High VIF Detected:</strong> Features with VIF &gt; 10 may have multicollinearity issues.</div>'
                html += '<table>'
                html += '<tr><th>Feature</th><th>VIF</th></tr>'
                for item in high_vif:
                    html += f'<tr><td>{item["feature"]}</td><td class="number">{item["VIF"]:.2f}</td></tr>'
                html += '</table>'
            else:
                html += '<div class="success-box">No high multicollinearity detected (all VIF &lt; 10).</div>'
            html += '</div></details>'

        # Recommendations (always open)
        html += '<details open>'
        html += '<summary>Recommendations</summary>'
        html += '<div class="section-content">'
        if report_data['recommendations']:
            for i, rec in enumerate(report_data['recommendations'], 1):
                html += f'<div class="recommendation"><strong>{i}.</strong> {rec}</div>'
        else:
            html += '<div class="info-box">No specific recommendations at this time.</div>'
        html += '</div></details>'

        html += """
    <hr style="margin-top: 30px; border: 0; border-top: 1px solid #dee2e6;">
    <p style="text-align: center; color: #666; font-size: 0.9em;">
        Report generated by Feature Engineering Toolkit
    </p>
</body>
</html>"""

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
