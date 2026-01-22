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
        """Export report as Markdown."""
        lines = []

        # Header
        lines.append(f"# Target Analysis Report")
        lines.append(f"\n**Generated**: {report_data['timestamp']}")
        lines.append(f"\n**Task Type**: {report_data['task'].upper()}")
        lines.append(f"\n**Target Column**: {report_data['task_info']['target_column']}")
        lines.append(f"\n---\n")

        # Task Info
        lines.append("## Task Information")
        lines.append(f"- **Data Type**: {report_data['task_info']['target_dtype']}")
        lines.append(f"- **Unique Values**: {report_data['task_info']['unique_values']}")
        lines.append(f"- **Missing Values**: {report_data['task_info']['missing_count']} ({report_data['task_info']['missing_percent']:.2f}%)")

        if report_data['task'] == 'classification':
            lines.append(f"- **Number of Classes**: {report_data['task_info']['class_count']}")
            lines.append(f"- **Classes**: {', '.join(map(str, report_data['task_info']['classes']))}")

        lines.append("\n---\n")

        # Distribution
        lines.append("## Distribution Analysis")
        if report_data['task'] == 'classification' and report_data['distribution']:
            lines.append("\n### Class Distribution")
            lines.append("| Class | Count | Percentage | Imbalance Ratio |")
            lines.append("|-------|-------|------------|-----------------|")
            for item in report_data['distribution']:
                lines.append(f"| {item['class']} | {item['count']} | {item['percentage']:.2f}% | {item['imbalance_ratio']:.2f} |")

            if report_data['imbalance']:
                lines.append(f"\n**Imbalance Severity**: {report_data['imbalance']['severity'].upper()}")
                lines.append(f"\n**Recommendation**: {report_data['imbalance']['recommendation']}")

        elif report_data['task'] == 'regression' and report_data['distribution']:
            lines.append("\n### Target Statistics")
            dist = report_data['distribution']
            lines.append(f"- **Mean**: {dist['mean']:.4f}")
            lines.append(f"- **Median**: {dist['median']:.4f}")
            lines.append(f"- **Std Dev**: {dist['std']:.4f}")
            lines.append(f"- **Skewness**: {dist['skewness']:.4f}")
            lines.append(f"- **Kurtosis**: {dist['kurtosis']:.4f}")

        lines.append("\n---\n")

        # Relationships
        if report_data['relationships']:
            lines.append("## Feature-Target Relationships")
            lines.append("\n### Top 10 Most Significant Features")
            lines.append("| Feature | Test Type | Statistic | P-Value | Significant |")
            lines.append("|---------|-----------|-----------|---------|-------------|")
            for item in report_data['relationships'][:10]:
                sig = "✓" if item['significant'] else "✗"
                lines.append(f"| {item['feature']} | {item['test_type']} | {item['statistic']:.4f} | {item['pvalue']:.4e} | {sig} |")
            lines.append("\n---\n")

        # Data Quality
        if report_data['data_quality']:
            lines.append("## Data Quality")
            quality = report_data['data_quality']

            if quality['missing_values']:
                lines.append("\n### Features with Missing Values")
                lines.append("| Feature | Missing Count | Missing % |")
                lines.append("|---------|---------------|-----------|")
                for feat, info in quality['missing_values'].items():
                    lines.append(f"| {feat} | {info['count']} | {info['percent']:.2f}% |")

            if quality['constant_features']:
                lines.append(f"\n### Constant Features: {', '.join(quality['constant_features'])}")

            if quality['leakage_suspects']:
                lines.append("\n### Potential Data Leakage")
                for suspect in quality['leakage_suspects']:
                    lines.append(f"- **{suspect['feature']}**: {suspect['reason']} (Severity: {suspect['severity']})")

            lines.append("\n---\n")

        # VIF
        if report_data['vif']:
            lines.append("## Multicollinearity (VIF)")
            lines.append("\n### Features with High VIF (>10)")
            high_vif = [item for item in report_data['vif'] if item['VIF'] > 10]
            if high_vif:
                lines.append("| Feature | VIF |")
                lines.append("|---------|-----|")
                for item in high_vif:
                    lines.append(f"| {item['feature']} | {item['VIF']:.2f} |")
            else:
                lines.append("No features with high multicollinearity detected.")
            lines.append("\n---\n")

        # Recommendations
        lines.append("## Recommendations")
        for rec in report_data['recommendations']:
            lines.append(f"- {rec}")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

    def _export_html(self, filepath: str, report_data: Dict[str, Any]) -> None:
        """Export report as HTML."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Target Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 40px auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #95a5a6; padding-bottom: 8px; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .info-box {{ background-color: #f8f9fa; border-left: 4px solid #3498db; padding: 15px; margin: 15px 0; }}
        .warning-box {{ background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 15px 0; }}
        .success-box {{ background-color: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 15px 0; }}
        .recommendation {{ padding: 10px; margin: 5px 0; background-color: #e9ecef; border-radius: 5px; }}
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

        # Add sections based on report data
        html += "<h2>Task Information</h2>"
        html += f"<p><strong>Data Type:</strong> {report_data['task_info']['target_dtype']}</p>"
        html += f"<p><strong>Unique Values:</strong> {report_data['task_info']['unique_values']}</p>"
        html += f"<p><strong>Missing:</strong> {report_data['task_info']['missing_count']} ({report_data['task_info']['missing_percent']:.2f}%)</p>"

        if report_data['task'] == 'classification':
            html += f"<p><strong>Classes:</strong> {len(report_data['task_info']['classes'])}</p>"

            if report_data['distribution']:
                html += "<h2>Class Distribution</h2>"
                html += "<table><tr><th>Class</th><th>Count</th><th>Percentage</th><th>Imbalance Ratio</th></tr>"
                for item in report_data['distribution']:
                    html += f"<tr><td>{item['class']}</td><td>{item['count']}</td><td>{item['percentage']:.2f}%</td><td>{item['imbalance_ratio']:.2f}</td></tr>"
                html += "</table>"

        elif report_data['task'] == 'regression' and report_data['distribution']:
            html += "<h2>Target Distribution</h2>"
            dist = report_data['distribution']
            html += f"<p><strong>Mean:</strong> {dist['mean']:.4f}</p>"
            html += f"<p><strong>Median:</strong> {dist['median']:.4f}</p>"
            html += f"<p><strong>Std Dev:</strong> {dist['std']:.4f}</p>"
            html += f"<p><strong>Skewness:</strong> {dist['skewness']:.4f}</p>"

        # Relationships
        if report_data['relationships']:
            html += "<h2>Feature-Target Relationships</h2>"
            html += "<table><tr><th>Feature</th><th>Test</th><th>Statistic</th><th>P-Value</th><th>Significant</th></tr>"
            for item in report_data['relationships'][:10]:
                sig = "✓" if item['significant'] else "✗"
                html += f"<tr><td>{item['feature']}</td><td>{item['test_type']}</td><td>{item['statistic']:.4f}</td><td>{item['pvalue']:.4e}</td><td>{sig}</td></tr>"
            html += "</table>"

        # Recommendations
        html += "<h2>Recommendations</h2>"
        for rec in report_data['recommendations']:
            html += f'<div class="recommendation">{rec}</div>'

        html += "</body></html>"

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html)
